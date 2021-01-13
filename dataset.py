from torch.utils.data import Dataset
import glob
import os
import utils.features as features
import torchaudio


def get_all_file_path(input_dir, file_extension='wav'):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp

def get_filename(input_filepath):
    return input_filepath.split('/')[-1]

def get_pure_filename(input_filepath):
    temp = input_filepath.split('/')[-1]
    return temp.split('.')[0]

def get_ravdess_property(input_filepath):
    pure_filename = get_pure_filename(input_filepath)
    idea = pure_filename.split('-')
    # 03-01-01-01-01-01-01
    return {'Modality': idea[0], 'VocalChannel': idea[1], 'Emotion': idea[2], 'EmotionalIntensity': idea[3],
            'Statement': idea[4], 'Repetition': idea[5], 'Actor': idea[6]}





class AudioDatasetType01(Dataset):
    def __init__(self, input_filepath, feature_config):
        self.input_filepath = input_filepath
        self.feature_config = feature_config
        self.file_list = get_all_file_path(input_filepath)


    def __getitem__(self, index):
        filepath = self.file_list[index]
        source, sr = torchaudio.load(filepath)
        if self.feature_config['spectrogram_type'] == 'spectrogram':
            waveform = features.extract_spectrogram(source=source,
                                                    sample_rate=sr,
                                                    n_fft=self.feature_config['n_fft'],
                                                    window_size=self.feature_config['window_size'],
                                                    window_stride=self.feature_config['window_stride'])
        elif self.feature_config['spectrogram_type'] == 'melspectrogram':
            waveform = features.extract_mel_spectrogram(source=source,
                                                        sample_rate=sr,
                                                        n_mels=self.feature_config['n_mels'],
                                                        n_fft=self.feature_config['n_fft'],
                                                        window_size=self.feature_config['window_size'],
                                                        window_stride=self.feature_config['window_stride'])







    def __len__(self):
        return len(self.file_list)
