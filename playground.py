import preprocessing.convert_wav_env as convert_wav_env
import load_dataset.data_split_functions as data_split_functions
import multiprocessing

# 1. start re-sample audio dataset
dest_dir = convert_wav_env.convert_wav_env_parallel_preprocess(input_dir='./dataset',
                                                               dest_dir='./dataset_resample',
                                                               target_sr=16000,
                                                               speed=1,
                                                               parallel=multiprocessing.cpu_count())

# 2. split train, test dataset
train_dataset, test_dataset = data_split_functions.split_train_test_file_list(root_dir=dest_dir,
                                                                              file_extension='wav')
print('{} : {}'.format(train_dataset[0], test_dataset[0]))