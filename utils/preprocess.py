import torch
import torchaudio
import glob
import os
import multiprocessing
import functools
from tqdm import tqdm

def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp

def get_pure_filename(input_filepath):
    temp = input_filepath.split('/')[-1]
    return temp.split('.')[0]

def get_filename(input_filepath):
    return input_filepath.split('/')[-1]

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def resampling(source, orig_freq=16000, new_freq=16000):
    resample = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
    return resample(source)

def find_short_data(source, sample_rate, duration):
    if duration is None:
        return True
    length = sample_rate*duration
    if len(source[0]) < length:
        return False
    else:
        return True

def extract_vad(source, sample_rate):
    vad = torchaudio.transforms.Vad(sample_rate=sample_rate)
    source = torch.flip(source, [0, 1])
    source = vad(source)
    source = torch.flip(source, [0, 1])
    return vad(source)

def preprocess(input_filepath, output_dir, duration=None):
    if not os.path.isdir(output_dir):
        create_directory(output_dir)
    source, sr = torchaudio.load(input_filepath)
    source = extract_vad(source, sr)
    if find_short_data(source, sr, duration=duration):
        torchaudio.save('{}{}'.format(output_dir, get_filename(input_filepath)),
                        src=source,
                        sample_rate=sr)
    else:
        return False
    return True

def parallel_preprocess(input_dir, output_dir, duration=None, parallel=None):
    file_list = get_all_file_path(input_dir, 'wav')
    with multiprocessing.Pool(parallel) as p:
        func = functools.partial(preprocess,
                                 output_dir=output_dir,
                                 duration=duration)
        output = list(tqdm(p.imap(func, file_list), total=len(file_list)))
        return output

def main():
    output = parallel_preprocess('../dataset/Audio_Speech_Actors_01-24/',
                                 '../dataset/speech_processed/',
                                 duration=None,
                                 parallel=multiprocessing.cpu_count())

if __name__ == '__main__':
    main()