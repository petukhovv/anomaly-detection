import math
import random
import numpy


class DatasetLoaderToDisk:
    def __init__(self, file, delimiter=','):
        self.csv = file
        self.csv_delimiter = delimiter

    def load(self, split_percent=0.1):
        test_sample_number = 940928
        train_sample_number = math.ceil(test_sample_number * split_percent)
        features_number = 11091

        dataset_test = numpy.memmap(
            'dataset_test.mymemmap', dtype='int', mode='w+', shape=(test_sample_number, features_number))
        dataset_train = numpy.memmap(
            'dataset_train.mymemmap', dtype='int', mode='w+', shape=(train_sample_number, features_number))

        random_numbers = set(random.sample(range(0, test_sample_number - 1), train_sample_number))

        with open(self.csv) as f:
            counter = 0
            train_counter = 0
            for line in f:
                arr = [int(x) for x in line.split(',')]
                if counter in random_numbers:
                    dataset_train[train_counter] = arr
                    print(str(train_counter + 1) + ' (' + str(counter + 1) + ') selected to train data')
                    train_counter += 1

                dataset_test[counter] = arr
                if counter % 50000 == 0:
                    dataset_test.flush()
                    dataset_train.flush()
                    print(str(counter + 1) + ' out of ' + str(test_sample_number) + ' finished')
                counter += 1

        dataset_test.flush()
        dataset_train.flush()
