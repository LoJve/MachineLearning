
from typing import List

class Preprocessing:

    @staticmethod
    def normalize_data(data):
        '''
            Normalize data
        '''
        min_value = data.min()
        max_value = data.max()
        normalize = (data - min_value) / (max_value - min_value)
        return normalize

    @staticmethod
    def split_and_normalize_data(data, percent = 0.7, is_normalized = False):
        '''
            Split and normalize data
        '''
        data_count, features_count = data.shape
        split_count = int(data_count * percent)
        train = data.iloc[0 : split_count, :]
        test = data.iloc[split_count : , :]
        train_X = train.iloc[:, 0 : features_count - 1]
        train_y = train.iloc[:, features_count - 1]
        test_X = test.iloc[:, 0 : features_count -1]
        test_y = test.iloc[:, features_count - 1]
        if is_normalized:
            train_X = Preprocessing.normalize_data(train_X)
            test_X = Preprocessing.normalize_data(test_X)
        return train_X, train_y, test_X, test_y



