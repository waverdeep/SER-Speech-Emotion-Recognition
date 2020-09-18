import functions.file_functions as file_functions
from sklearn.model_selection import train_test_split


def split_train_test_file_list(root_dir):
    file_list = file_functions.get_all_file_path(root_dir)
    file_label = file_functions.get_emotion_type(file_list)

    x_train, x_test, y_train, y_test = train_test_split(file_list, file_label, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
