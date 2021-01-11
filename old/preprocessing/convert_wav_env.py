import os
import glob
import functools
import sox
import multiprocessing
from tqdm import tqdm


def get_all_file_path(input_dir, file_extension):
    return glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)


def preprocess(data, dest_dir, target_sr=16000, speed=1):
    output_file_path = os.path.join(dest_dir, '/'.join(data.split('/')[2:-1]))
    os.makedirs(output_file_path, exist_ok=True)
    if not os.path.exists(output_file_path) or True:
        output = sox.Transformer().speed(factor=speed).convert(target_sr)
        output.build(data, os.path.join(dest_dir, '/'.join(data.split('/')[2:])))
        return 0


def convert_wav_env_parallel_preprocess(input_dir, dest_dir, target_sr=16000, speed=1, parallel=None):
    dataset = get_all_file_path(input_dir, 'wav')
    with multiprocessing.Pool(parallel) as p:
        func = functools.partial(preprocess,
                                 dest_dir=dest_dir,
                                 target_sr=target_sr,
                                 speed=speed)
        output = list(tqdm(p.imap(func, dataset), total=len(dataset)))
        return dest_dir
