import math
import numpy


class DatasetLoader:
    def __init__(self, file, delimiter=','):
        self.csv = file
        self.csv_delimiter = delimiter

    def load(self, split_percent=0.1):
        test_sample_number = 940928
        train_sample_number = math.ceil(test_sample_number * split_percent)
        features_number = 11091

        print('Start dataset load')
        dataset_test = numpy.memmap(
            'dataset_test.mymemmap', dtype='int', mode='r', shape=(test_sample_number, features_number))
        dataset_train = numpy.memmap(
            'dataset_train.mymemmap', dtype='int', mode='r', shape=(train_sample_number, features_number))
        print('Link to dataset from disk set')

        if len(dataset_test) == 0:
            return None, None, None

        return dataset_train, dataset_test, features_number
