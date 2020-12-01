import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, load_checkpoint, save_checkpoint, set_init_dict
import glob

torch.backends.cudnn.benchmark = True
import librosa
import numpy as np

import os
import sys


TTS_PATH = "../glowtts/TTS-dev/"
sys.path.append(TTS_PATH) 
from TTS.tts.utils.speakers import save_speaker_mapping, load_speaker_mapping
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.speaker_encoder.model import SpeakerEncoder


USE_CUDA = True

MODEL_RUN_PATH = "../glowtts/checkpoint-SpeakerEncoder/"
MODEL_PATH_SE = MODEL_RUN_PATH + "320k.pth.tar"
CONFIG_PATH_SE = MODEL_RUN_PATH + "config.json"


c_se = load_config(CONFIG_PATH_SE)

se_ap = AudioProcessor(**c_se['audio'])

se_model = SpeakerEncoder(**c_se.model)
se_model.load_state_dict(torch.load(MODEL_PATH_SE, map_location=torch.device('cpu'))['model'])
se_model.eval()
if USE_CUDA:
    se_model.cuda()

def calc_emb(y):
  y_16 = librosa.resample(y, 22050, se_ap.sample_rate)
  #y_16, sr = librosa.load(wav_file, sr=se_ap.sample_rate)
  mel_spec = se_ap.melspectrogram(y_16)
  mel_spec = torch.FloatTensor(mel_spec[None, :, :]).transpose(1, 2)
  # print(mel_spec.shape)
  if USE_CUDA:
      mel_spec = mel_spec.cuda()
  return se_model.compute_embedding(mel_spec).cpu().detach().numpy().reshape(-1)

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def train(rank, a, h, speaker_mapping=None):
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_filelist = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir, use_speaker_embedding=h.use_speaker_embedding,
                          speaker_mapping=speaker_mapping)

    validation_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False,
                              sampler=None,
                              batch_size=a.batch_size,
                              pin_memory=True,
                              drop_last=True)
    min_loss = 0
    best_checkpoint = ''
    # list all checkpoints
    cp_list = glob.glob(os.path.join(a.checkpoint_path, 'g_*'))

    for checkpoint_g in cp_list:
        generator = Generator(h).to(device)
        state_dict_g = load_checkpoint(checkpoint_g, device)
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()

        total_checkpoint_loss = 0

        torch.cuda.empty_cache()
        with torch.no_grad():
            for i, batch in enumerate(validation_loader):
                x, y, _, _, speaker_embedding = batch
                if speaker_embedding is not None:
                    speaker_embedding = speaker_embedding.to(device)

                y = y.unsqueeze(1).to(device)
                y_g_hat = generator(x.to(device), speaker_embedding)
                speaker_embedding = speaker_embedding.cpu().numpy()
                # print(speaker_embedding.size(1))
                y_g_hat = y_g_hat.cpu().numpy()
                for j in range(len(y_g_hat)):
                    emb = calc_emb(y_g_hat[j].reshape(-1))
                    if speaker_embedding is None or speaker_embedding.shape[-1] == 0:
                        emb2 = calc_emb(y[j].reshape(-1))
                    else:
                        emb2 = speaker_embedding[j]
                    total_checkpoint_loss += cosine_similarity(emb, emb2)

        print("Checkpoint", checkpoint_g, "L1 Loss:", total_checkpoint_loss/len(validset))
        if total_checkpoint_loss > min_loss:
            min_loss = total_checkpoint_loss
            best_checkpoint = checkpoint_g
        
    # cp best checkpoint 
    copyfile(best_checkpoint, best_checkpoint.replace("g_", "g_best_"))
    do_best_checkpoint = best_checkpoint.replace('g_', 'do_')
    copyfile(do_best_checkpoint, do_best_checkpoint.replace("do_", "do_best_"))

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--speakers_json', default=None, type=str)
    parser.add_argument('--batch_size', default=15, type=int)
    # for train script compatibility
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')

    

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()
    
    if a.speakers_json:
      with open(a.speakers_json) as f:
            speaker_mapping = json.load(f)
    else:
        speaker_mapping = None

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass


    train(0, a, h, speaker_mapping)


if __name__ == '__main__':
    main()
