import preprocessing.convert_wav_env as convert_wav_env
import load_dataset.data_split_functions as data_split_functions
import multiprocessing
import load_dataset.AudioDataset as AudioDataset
from torch.utils.data import DataLoader

# 1. start re-sample audio dataset
dest_dir = convert_wav_env.convert_wav_env_parallel_preprocess(input_dir='./dataset',
                                                               dest_dir='./dataset_resample',
                                                               target_sr=16000,
                                                               speed=1,
                                                               parallel=multiprocessing.cpu_count())

# 2. split train, test dataset
train_filelist, test_filelist = data_split_functions.split_train_test_file_list(root_dir=dest_dir,
                                                                                file_extension='wav')

train_dataset = AudioDataset.AudioDataset(dataset=train_filelist)
test_dataset = AudioDataset.AudioDataset(dataset=test_filelist)

train_dataloader = DataLoader(train_filelist, batch_size=32, shuffle=True, num_workers=multiprocessing.cpu_count())
test_dataloader = DataLoader(test_filelist, batch_size=32, shuffle=False, num_workers=multiprocessing.cpu_count())
