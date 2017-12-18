import math

import numpy as np


class DatasetLoader:
    def __init__(self, file, delimiter=','):
        self.csv = file
        self.csv_delimiter = delimiter

    def load(self, split_percent=0.1):
        dataset = np.loadtxt(self.csv, delimiter=self.csv_delimiter)
        bound = math.ceil(len(dataset) * split_percent)
        train_data = dataset[:bound]
        test_data = dataset[bound:]

        if len(test_data) == 0:
            return None, None, None

        features_number = len(test_data[0])

        x_train = np.reshape(train_data, (len(train_data), features_number))
        x_test = np.reshape(test_data, (len(test_data), features_number))

        return x_train, x_test, features_number
