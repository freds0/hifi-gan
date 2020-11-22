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
    min_loss = 99999999999999
    best_checkpoint = ''
    # list all checkpoints
    cp_list = glob.glob(os.path.join(a.checkpoint_path, 'g_*'))

    for checkpoint_g in cp_list:
        generator = Generator(h).to(device)
        mpd = MultiPeriodDiscriminator().to(device)
        msd = MultiScaleDiscriminator().to(device)
        state_dict_do = load_checkpoint(checkpoint_g.replace('g_','do_'), device)
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])

        state_dict_g = load_checkpoint(checkpoint_g, device)
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        msd.eval()
        mpd.eval()

        total_checkpoint_loss = 0

        torch.cuda.empty_cache()
        with torch.no_grad():
            for i, batch in enumerate(validation_loader):
                x, y, _, _, speaker_embedding = batch
                if speaker_embedding is not None:
                    speaker_embedding = speaker_embedding.to(device)

                y = y.unsqueeze(1).to(device)
                y_g_hat = generator(x.to(device), speaker_embedding)
                # print(y_g_hat.shape, y.shape)
                y_mel =  mel_spectrogram(y.squeeze(1).to(device), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                h.fmin, h.fmax_for_loss, center=False)
                
                
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                h.fmin, h.fmax_for_loss, center=False)

                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel)
                
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel


                total_checkpoint_loss += loss_gen_all.item()

            
        print("Checkpoint", checkpoint_g, "L1 Loss:", total_checkpoint_loss)
        if total_checkpoint_loss < min_loss:
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
