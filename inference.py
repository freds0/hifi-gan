from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
import torchaudio
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x, sr=None):
    
    
    if h.VOC_sampling_rate != h.TTS_sampling_rate:
        if sr != h.TTS_sampling_rate:
            audio_resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=h.TTS_sampling_rate, resampling_method='sinc_interpolation')
            x = audio_resample(x)
        mel = mel_spectrogram(x, h.n_fft, h.num_mels, h.TTS_sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
        mel = torch.nn.functional.interpolate(mel.unsqueeze(0), scale_factor=(1, int(h.VOC_sampling_rate/h.TTS_sampling_rate)), mode='bilinear', align_corners=True).squeeze(0)
    else:
        if sr != h.VOC_sampling_rate:
            audio_resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=h.VOC_sampling_rate, resampling_method='sinc_interpolation')
            x = audio_resample(x)
        mel = mel_spectrogram(x, h.n_fft, h.num_mels, h.VOC_sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
    return mel


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filename))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0), sr)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated.wav')
            write(output_file, h.VOC_sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

