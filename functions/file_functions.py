import os
import glob


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
