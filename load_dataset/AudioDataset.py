import torch
import torch.utils
import pandas as pd
import functions.file_functions as file_functions
import functions.spectrogram_functions as spectrogram_functions


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.dataset = []
        file_list = file_functions.get_all_file_path(root_dir)
        for file in file_list:
            temp = [file, file_functions.get_emotion_type(file)]
            self.dataset.append(temp)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = torch.from_numpy(spectrogram_functions.scipy_log_spectrogram(self.dataset[idx][0],
                                                                         sr=16000,
                                                                         res_type='kaiser_fast',
                                                                         duration=2.8))
        y = self.dataset[idx][1]
        return x, y


