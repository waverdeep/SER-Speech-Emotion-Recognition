import os
import glob
import pandas
import numpy as np
import librosa
import librosa.display
import scipy.io.wavfile
from scipy.fftpack import fft
from scipy import signal
import sys
import matplotlib.pyplot as plt
import scipy.io


def load_wav_librosa(filepath, sr=16000, res_type='kaiser_fast', duration=2.8):
    samples, sample_rate = librosa.load(filepath, sr=sr, res_type=res_type, duration=duration)
    return samples, sample_rate


def log_spectrogram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


def scipy_log_spectrogram(filepath, sr=16000, res_type='kaiser_fast', duration=2.8):
    samples, sample_rate = load_wav_librosa(filepath, sr=sr, res_type=res_type, duration=duration)
    freqs, times, spec = log_spectrogram(samples, sample_rate)
    spec = np.expand_dims(spec, axis=0)
    print(spec.shape)
    return freqs, times, spec


def scipy_log_spectrogram_plot(freqs, times, spectrogram):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    plt.yticks(freqs[::16])
    plt.xticks(times[::16])
    plt.title('Spectrogram')
    plt.ylabel('Freqs in Hz')
    plt.xlabel('Seconds')


def mel_power_spectrogram(filepath, sr=16000, res_type='kaiser_fast', duration=2.8):
    samples, sample_rate = load_wav_librosa(filepath, sr=sr, res_type=res_type, duration=duration)
    freqs, times, spectrogram = log_spectrogram(samples, sample_rate)
    # Plotting Mel Power Spectrogram
    S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


def mel_power_spectrogram_plot(log_S, sample_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.title('Mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()


def delta2_mfcc(filepath, sr=16000, res_type='kaiser_fast', duration=2.8):
    samples, sample_rate = librosa.load(filepath, sr=sr, res_type=res_type, duration=duration)
    freqs, times, spectrogram = log_spectrogram(samples, sample_rate)
    # Plotting Mel Power Spectrogram
    S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    # Plotting MFCC
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    # Let's pad on the first and second deltas while we're at it
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return delta2_mfcc


def delta2_mfcc_plot(delta2_mfcc):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(delta2_mfcc)
    plt.ylabel('MFCC coeffs')
    plt.xlabel('Time')
    plt.title('MFCC')
    plt.colorbar()
    plt.tight_layout()

