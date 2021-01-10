import librosa
import functions.file_functions as file_functions
import numpy as np


def mel_spectrogram(data, sr, n_mels, n_fft, hop_length):
    return librosa.feature.melspectrogram(data, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

def extract_mel_spectrogram(filepath, sample_rate, n_mels, n_fft=2048, hop_length=512, duration=None, res_type='kaiser_best'):
    data, sr = file_functions.load_wav(filepath, sample_rate=sample_rate, duration=duration, res_type=res_type)
    return mel_spectrogram(data, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

def extract_power_mel_spectrogram(filepath, sample_rate, n_mels, n_fft=2048, hop_length=512, duration=None, res_type='kaiser_best'):
    data, sr = file_functions.load_wav(filepath, sample_rate=sample_rate, duration=duration, res_type=res_type)
    melspectrogram =  mel_spectrogram(data, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    return librosa.amplitude_to_db(melspectrogram, ref=np.max)

