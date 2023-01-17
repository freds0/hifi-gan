import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import torchaudio

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, TTS_sampling_rate, fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None, use_interpolation=True):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.TTS_sampling_rate = TTS_sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.cached_wav_input = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.use_interpolation = use_interpolation
        self.audio_resample = torchaudio.transforms.Resample(orig_freq=self.sampling_rate, new_freq=self.TTS_sampling_rate, resampling_method='sinc_interpolation')
        self.resample_factor = float(self.sampling_rate)/float(self.TTS_sampling_rate)

    def __getitem__(self, index):
        filename = self.audio_files[index]

        if self._cache_ref_count == 0:
            try:
                audio, sampling_rate = load_wav(filename)
                audio = audio / MAX_WAV_VALUE
                #if not self.fine_tuning:
                audio = normalize(audio) * 0.95
                # convert to torch
                audio = torch.FloatTensor(audio)
                audio = audio.unsqueeze(0)
            except:
                # if file dont exist or is corrupted, select other sample
                print("WARNING: The file", filename, "Don't exist or is corrupted, please check this. We selected other sample ...")
                return self.__getitem__(random.randint(0, self.__len__()))

            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            if self.sampling_rate != self.TTS_sampling_rate and not self.fine_tuning:
                audio_frames = audio.size(1) 
                # if odd turn even
                if audio_frames % 2 != 0:
                    audio = audio[:, :audio_frames-1]

                # resample input sample to TTS sampling_rate
                audio_input = self.audio_resample(audio)
            else:
                audio_input = None

            self.cached_wav = audio
            self.cached_wav_input = audio_input

            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            audio_input = self.cached_wav_input
            self._cache_ref_count -= 1

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    # generate a random even number
                    while audio_start % 2 != 0:
                        audio_start = random.randint(0, max_audio_start)

                    audio_end = audio_start+self.segment_size
                    audio = audio[:, audio_start:audio_end]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
                
                if self.sampling_rate != self.TTS_sampling_rate:
                    if audio_input.size(1) >= self.segment_size/self.resample_factor:
                        audio_start = int(audio_start/self.resample_factor)
                        audio_end = int(audio_end/self.resample_factor)
                        audio_input = audio_input[:, audio_start:audio_end]
                    else:
                        audio_input = torch.nn.functional.pad(audio_input, (0, int(self.segment_size/self.resample_factor) - audio_input.size(1)), 'constant')

            if self.sampling_rate != self.TTS_sampling_rate:               
                mel = mel_spectrogram(audio_input, self.n_fft, self.num_mels,
                                  self.TTS_sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
                # upsample TTS spec to vocoder
                if self.use_interpolation:
                    mel = torch.nn.functional.interpolate(mel.unsqueeze(0), scale_factor=(1, self.resample_factor), mode='bilinear', align_corners=True, recompute_scale_factor=True).squeeze(0)

            else:
                mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                    self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                    center=False)

        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)#.transpose(-2,-1)
            '''if mel.size(-1) == self.num_mels:
                mel = mel.transpose(-2,-1)'''
            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            # upsample TTS spec to vocoder
            if self.use_interpolation:
                mel = torch.nn.functional.interpolate(mel.unsqueeze(0), scale_factor=(1, self.resample_factor), mode='bilinear', align_corners=True, recompute_scale_factor=True).squeeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 5)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]

                mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
