import pandas as pd
from config import TRAIN_TEST_SPLIT_RATIO

def train_test_split(data_path, train_ratio=TRAIN_TEST_SPLIT_RATIO):
    data = pd.read_csv(data_path)
    count = len(data)
    train_data = data[:int(train_ratio*count)]
    test_data = data[int(train_ratio*count):]
    return train_data, test_data