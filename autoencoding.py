import math
import funcy
import json
import struct
import numpy as np

from lib.Autoencoder import Autoencoder
from lib.DatasetLoader import DatasetLoader
from lib.helpers.TimeLogger import TimeLogger


def binary_write(differences, features_number, output_file):
    chunking_time_logger = TimeLogger()
    differences = differences.flatten('F')
    differences = np.append(differences, features_number)
    differences = struct.pack('=%df' % differences.size, *differences)

    chunk_size = 10000000
    difference_chunks = funcy.chunks(chunk_size, differences)
    print('Chunking finished. Time: ' + str(chunking_time_logger.finish()))

    chunk_counter = 1
    for difference_chunk in difference_chunks:
        with open(output_file, 'ab') as f:
            difference_chunk_time_logger = TimeLogger()
            f.write(difference_chunk)
            print('Write difference chunk ' + str(chunk_counter) + ' is done. Time: ' +
                  str(difference_chunk_time_logger.finish()))
            chunk_counter += 1


def ascii_write(differences, output_file):
    with open(output_file, 'w') as f:
        f.write(json.dumps(differences))


def autoencoding(dataset_file, split_percent, encoding_dim_percent, output_file=None, full_differences=None):
    time_logger = TimeLogger()
    data = DatasetLoader(dataset_file).load(split_percent=split_percent)
    (_, _, features_number) = data
    encoding_dim = math.ceil(features_number * encoding_dim_percent)
    print('Dataset loaded. Time: ' + str(time_logger.finish()))

    time_logger = TimeLogger()
    autoencoder = Autoencoder(features_number, encoding_dim, data)
    autoencoder.print_model_summary()
    autoencoder.fit()
    print('Autoencoder fit finished. Time: ' + str(time_logger.finish()))

    time_logger = TimeLogger()
    autoencoder.predict()
    print('Autoencoder predict finished. Time: ' + str(time_logger.finish()))

    time_logger = TimeLogger()
    differences = autoencoder.calc_differences(full_differences)
    print('Calculate differences finished. Time: ' + str(time_logger.finish()))

    if not full_differences:
        differences = sorted(enumerate(differences), key=lambda tup: tup[1], reverse=True)

    if not output_file:
        return differences

    time_logger = TimeLogger()

    if full_differences:
        binary_write(differences, features_number, output_file)
    else:
        ascii_write(differences, output_file)

    print('Write differences finished. Time: ' + str(time_logger.finish()))
