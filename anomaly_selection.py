import numpy as np
import struct
import json

from sklearn.cluster import DBSCAN

from lib.helpers.TimeLogger import TimeLogger


def binary_read(differences_file):
    differences_read_logger = TimeLogger()
    with open(differences_file, 'rb') as f:
        buffer = f.read(4)
        differences = []
        chunk_counter = 0
        log_write_per_chunk_number = 10000000
        chunks_time_logger = TimeLogger()
        while buffer:
            differences.append(struct.unpack('=f', buffer))
            buffer = f.read(4)
            if (chunk_counter + 1) % log_write_per_chunk_number == 0:
                print(
                    str(chunk_counter + 1) + ' chunks is read and unpacked. Time: ' + str(chunks_time_logger.finish()))
                chunks_time_logger = TimeLogger()
            chunk_counter += 1
        print(str(chunk_counter) + ' chunks is read and unpacked. Time: ' + str(chunks_time_logger.finish()))
    print('Differences read finished. Time: ' + str(differences_read_logger.finish()))

    transformation_logger = TimeLogger()
    differences = np.array(differences)
    features_number = int(differences[-1])
    differences = differences[:-1].reshape(int(len(differences) / features_number), features_number)
    print('Differences transformation finished. Time: ' + str(transformation_logger.finish()))

    return differences


def ascii_read(differences_file):
    differences_read_logger = TimeLogger()
    with open(differences_file) as f:
        differences = json.loads(f.read())
        differences_cleared = []
        for difference in differences:
            differences_cleared.append(difference[1])

    print('Differences read finished. Time: ' + str(differences_read_logger.finish()))

    return differences_cleared


def anomaly_selection(files_map_file, anomalies_output_file, differences_file=None, differences=None):
    if differences_file:
        differences = binary_read(differences_file)
        # differences = ascii_read(differences_file)

    dbscan_time_logger = TimeLogger()
    print(differences.shape)
    labels = DBSCAN(eps=3, min_samples=5, algorithm='ball_tree').fit_predict(differences)
    anomaly_indexes = [i for i, x in enumerate(labels) if x == -1]
    print('DBSCAN finished its work. Time: ' + str(dbscan_time_logger.finish()))

    anomalies_write_time_logger = TimeLogger()
    with open(files_map_file) as files_map_file_descriptor:
        files_map = json.loads(files_map_file_descriptor.read())
        anomaly_files = []
        for anomaly_index in anomaly_indexes:
            anomaly_files.append(files_map[anomaly_index])

        with open(anomalies_output_file, 'w') as anomalies_output_file_descriptor:
            anomalies_output_file_descriptor.write(json.dumps(anomaly_files))

    print('Anomaly list written. Time: ' + str(anomalies_write_time_logger.finish()))

    return len(anomaly_files)
