import os
import glob
import librosa

# find all dataset filepath
def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp


# get pure filename
def get_filename(input_filepath):
    temp = input_filepath.split('/')[-1]
    return temp.split('.')[0]


def get_emotion_type(input_filepath):
    temp = input_filepath.split('/')[-1]
    temp = temp.split('.')[0]
    return temp.split('-')[2]

def load_wav(filepath, sample_rate=16000, duration=None, res_type='kaiser_best'):
    return librosa.load(filepath, sr=sample_rate, duration=duration, res_type=res_type)
