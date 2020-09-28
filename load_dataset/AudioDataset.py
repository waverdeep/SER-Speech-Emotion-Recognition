import torch
import torch.utils
import pandas as pd
import functions.file_functions as file_functions
import functions.spectrogram_functions as spectrogram_functions
import numpy as np


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_rate=16000, res_type='kaiser_fast', duration=2.8):
        self.sample_rate = sample_rate
        self.res_type = res_type
        self.duration = duration
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # duration 설정을 통해서 서로 다른 길이의 음원들을 모두 같은 길이로 맞춰줌
        # 추후에 다른 방법으로 처리해야 할 듯 ( 서로 다른 길이의 음원에 silence 한 부분을 제거하고,
        #                              최대 사이즈를 정의한 다음 빈 부분에 대해서는 default 값을 채우는 방법도 생각해 보아야 함)
        x = self.get_mel_spectrogram_format(self, filepath=self.dataset[idx], duration=self.duration)
        y = self.dataset[idx][1]
        return x, y

    def get_mel_spectrogram_format(self, filepath, duration):
        signal, sample_rate = spectrogram_functions.load_wav_librosa(self.filepath, 16000)
        signal, index = spectrogram_functions.trim_silence(signal)
        signal = spectrogram_functions.noise_reduction(signal)

        if len(signal) > duration:
            signal = signal[0:duration]
        elif duration > len(signal):
            max_offset = duration - len(signal)
            signal = np.pad(signal, (0, max_offset), "constant")

        signal = spectrogram_functions.mel_spectrogram(signal)
        signal = np.expand_dims(signal, axis=-2)

        return signal


