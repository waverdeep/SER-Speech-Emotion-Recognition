import torch
import models.network as network
import utils.features as features
import dataset
from dataset import AudioDatasetType01
import argparse
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition')
    parser.add_argument('--input_config', required=False,
                        default='./configs/configs_spectrogram.json', help="input config path")
    args = parser.parse_args()
    return args

def cuda_check():
    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
    return is_cuda

def read_input_config(input_filepath):
    with open(input_filepath, 'r') as f:
        json_data = json.load(f)
    return json_data

def visualize_dataset(input_dataset, count=10):
    for i in range(len(input_dataset)):
        x, y = input_dataset[i]
        print(i, x.shape, y)
        plt.pcolor(x)
        plt.show()

        if i == count:
            break

def main(args):
    configs = read_input_config(args.input_config)
    is_cuda = cuda_check()
    print("cuda status : {}".format(is_cuda))
    train_file_list, valid_file_list, test_file_list = dataset.unweighted_split_train_test_file_list(
        input_dir=configs['input_dir'],
        file_extension=configs['input_file_extension'],
        test_size=0.2
    )
    print('train_list : {}'.format(len(train_file_list)))
    print('valid_list : {}'.format(len(valid_file_list)))
    print('test_list : {}'.format(len(test_file_list)))

    # data load
    train_dataset = AudioDatasetType01(input_file_list=train_file_list,
                                       feature_config=configs['feature_config'])
    valid_dataset = AudioDatasetType01(input_file_list=valid_file_list,
                                       feature_config=configs['feature_config'])
    test_dataset = AudioDatasetType01(input_file_list=test_file_list,
                                       feature_config=configs['feature_config'])

    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    # model

    # optimizer





if __name__ == '__main__':
    args = get_args()
    main(args)