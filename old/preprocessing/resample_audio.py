import old.preprocessing.convert_wav_env as convert_wav_env
import multiprocessing

# 1. start re-sample audio dataset
dest_dir = convert_wav_env.convert_wav_env_parallel_preprocess(input_dir='./dataset',
                                                               dest_dir='./dataset_resample',
                                                               target_sr=16000,
                                                               speed=1,
                                                               parallel=multiprocessing.cpu_count())
