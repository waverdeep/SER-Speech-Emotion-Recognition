import functions.file_functions as file_functions
from sklearn.model_selection import train_test_split


def split_train_test_file_list(root_dir, file_extension='wav', test_size=0.2, random_state=42):
    file_list = file_functions.get_all_file_path(root_dir, file_extension)

    train_dataset, test_dataset = train_test_split(file_list, test_size=test_size, random_state=random_state)
    return train_dataset, test_dataset
