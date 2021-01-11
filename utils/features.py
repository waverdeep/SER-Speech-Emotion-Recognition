import torchaudio
import torch.nn as nn

def extract_spectrogram(source, sample_rate,n_fft=None, window_size=0.025, window_stride=0.01):
    if n_fft is None:
        n_fft = int(round(sample_rate * window_size))
    win_length = int(round(sample_rate * window_size))
    hop_length = int(round(sample_rate * window_stride))

    spectrogram = nn.Sequential(
        torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length),
        torchaudio.transforms.AmplitudeToDB()
    )
    return spectrogram(source[0])

def extract_mel_spectrogram(source, sample_rate, n_mels=80, n_fft=None, window_size=0.025, window_stride=0.01):
    if n_fft is None:
        n_fft = int(round(sample_rate * window_size))
    win_length = int(round(sample_rate * window_size))
    hop_length = int(round(sample_rate * window_stride))
    mel_spectrogram = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels),
        torchaudio.transforms.AmplitudeToDB()
    )
    return mel_spectrogram(source[0])

def extract_mfcc(source, sample_rate, n_mfcc, log_mels=True, melkargs=None):
    mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, log_mels=log_mels, melkwargs=melkargs)
    return mfcc(source[0])


