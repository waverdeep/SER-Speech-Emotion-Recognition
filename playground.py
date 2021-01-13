import torch
import models.network as network
import utils.features as features
import dataset
from dataset import AudioDatasetType01
import argparse
import json

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



if __name__ == '__main__':
    args = get_args()
    main(args)