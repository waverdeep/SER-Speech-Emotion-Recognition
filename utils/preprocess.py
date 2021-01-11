import torch
import torchaudio
import torch.nn as nn

def resampling(source, orig_freq=16000, new_freq=16000):
    resample = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
    return resample(source)

def remove_short_data():
    pass


def extract_vad(source, sample_rate):
    vad = torchaudio.transforms.Vad(sample_rate=sample_rate)
    source = torch.flip(source, [0, 1])
    source = vad(source)
    source = torch.flip(source, [0, 1])
    return vad(source)

