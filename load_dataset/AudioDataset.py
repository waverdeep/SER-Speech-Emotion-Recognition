import torch
import torch.utils
import pandas as pd
import functions.file_functions as file_functions
import functions.spectrogram_functions as spectrogram_functions


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_rate=16000, res_type='kaiser_fast', duration=2.8):
        self.sample_rate = sample_rate
        self.res_type = res_type
        self.duration = duration
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = spectrogram_functions.scipy_log_spectrogram(self.dataset[idx][0],
                                                        sr=self.sample_rate,
                                                        res_type=self.res_type,
                                                        duration=self.duration)
        # duration 설정을 통해서 서로 다른 길이의 음원들을 모두 같은 길이로 맞춰줌
        # 추후에 다른 방법으로 처리해야 할 듯 ( 서로 다른 길이의 음원에 silence 한 부분을 제거하고,
        #                              최대 사이즈를 정의한 다음 빈 부분에 대해서는 default 값을 채우는 방법도 생각해 보아야 함)

        y = self.dataset[idx][1]
        return x, y


