import math

import numpy as np


class DatasetLoader:
    def __init__(self, file, delimiter=','):
        self.csv = file
        self.csv_delimiter = delimiter

    def load(self, split_percent=0.1):
        dataset = np.loadtxt(self.csv, delimiter=self.csv_delimiter)
        bound = math.ceil(len(dataset) * split_percent)

        if len(dataset) == 0:
            return None, None, None

        features_number = len(dataset[0])

        return dataset[bound:], dataset, features_number
