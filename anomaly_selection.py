import numpy as np
import struct
import json

from sklearn.cluster import DBSCAN

from lib.helpers.TimeLogger import TimeLogger


def binary_read(differences_file):
    differences_read_logger = TimeLogger(task_name='Differences read')
    with open(differences_file, 'rb') as f:
        buffer = f.read(4)
        differences = []
        chunk_counter = 0
        log_write_per_chunk_number = 10000000
        chunks_time_logger = TimeLogger(task_name='Read and unpack %d chunks' % chunk_counter)
        while buffer:
            differences.append(struct.unpack('=f', buffer))
            buffer = f.read(4)
            if (chunk_counter + 1) % log_write_per_chunk_number == 0:
                chunks_time_logger.finish()
                chunks_time_logger = TimeLogger(task_name='Read and unpack %d chunks' % chunk_counter)
            chunk_counter += 1
        chunks_time_logger.finish()
    differences_read_logger.finish()

    transformation_logger = TimeLogger(task_name='Differences transformation')
    differences = np.array(differences)
    features_number = int(differences[-1])
    differences = differences[:-1].reshape(int(len(differences) / features_number), features_number)
    transformation_logger.finish()

    return differences


def ascii_read(differences_file):
    differences_read_logger = TimeLogger(task_name='Differences read')
    with open(differences_file) as f:
        differences = json.loads(f.read())

        difference_indexes = []
        difference_values = []
        for difference in differences:
            difference_indexes.append(difference[0])
            difference_values.append(difference[1])

    differences_read_logger.finish()

    return difference_indexes, difference_values


def dbscan_anomaly_selection(differences):
    dbscan_time_logger = TimeLogger('DBSCAN its work')
    labels = DBSCAN(eps=3, min_samples=5, metric='euclidean').fit_predict(differences)
    anomaly_indexes = [i for i, x in enumerate(labels) if x == -1]
    dbscan_time_logger.finish(full_finish=True)

    return anomaly_indexes


def three_sigma_anomaly_selection(differences):
    three_sigma_time_logger = TimeLogger('3-sigma anomaly selection')

    difference_indexes, difference_values = differences
    mean = np.mean(difference_values)
    std_deviation = np.std(difference_values)
    left_bound_3_sigma = mean - 3 * std_deviation
    right_bound_3_sigma = mean + 3 * std_deviation

    anomalies = []
    for i, x in enumerate(difference_values):
        if x < left_bound_3_sigma or x > right_bound_3_sigma:
            anomalies.append((difference_indexes[i], difference_values[i]))

    three_sigma_time_logger.finish(full_finish=True)

    return anomalies


def anomaly_selection(files_map_file, anomalies_output_file, use_dbscan, differences_file=None, differences=None):
    if differences_file:
        if use_dbscan:
            differences = binary_read(differences_file)
        else:
            differences = ascii_read(differences_file)

    if use_dbscan:
        anomalies = dbscan_anomaly_selection(differences)
    else:
        anomalies = three_sigma_anomaly_selection(differences)

    anomalies_write_time_logger = TimeLogger('Anomaly list write')
    with open(files_map_file) as files_map_file_descriptor:
        files_map = json.loads(files_map_file_descriptor.read())
        anomaly_files = []
        if use_dbscan:
            for anomaly_index in anomalies:
                anomaly_files.append(files_map[anomaly_index])
        else:
            for anomaly_index, anomaly_value in anomalies:
                anomaly_files.append((files_map[anomaly_index], anomaly_value))

        with open(anomalies_output_file, 'w') as anomalies_output_file_descriptor:
            anomalies_output_file_descriptor.write(json.dumps(anomaly_files))

    anomalies_write_time_logger.finish(full_finish=True)

    return len(anomaly_files)
