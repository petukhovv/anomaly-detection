import argparse
import os

from autoencoding import autoencoding
from anomaly_selection import anomaly_selection


parser = argparse.ArgumentParser()
parser.add_argument('--stage', '-s', choices=['autoencoding', 'anomaly_selection'])
parser.add_argument('--use_dbscan', action='store_true',
                    help='whether to use dbscan (high memory usage);'
                         'if not, then will use simple euclidean distance between input and output vector')

# Autoencoding stage params
parser.add_argument('--dataset', '-f', nargs=1, type=str, help='path to dataset file (csv format with colon delimiter)')
parser.add_argument('--split_percent', nargs=1, type=float, help='dataset train/test split percent')
parser.add_argument('--encoding_dim_percent', nargs=1, type=float,
                    help='encoding dim percent (towards features number)')
parser.add_argument('--differences_output_file', nargs=1, type=str,
                    help='path to file with input-decoded difference')

# Anomaly selection stage params
parser.add_argument('--differences_file', nargs=1, type=str,
                    help='path to file with distance vectors (obtained by autoencoder)')
parser.add_argument('--files_map_file', nargs=1, type=str,
                    help='path to file with map dataset indexes and ast file paths')
parser.add_argument('--anomalies_output_file', '-o', nargs=1, type=str,
                    help='path to file, which will contain anomaly list (as paths to AST code snippets)')

args = parser.parse_args()
stage = args.stage

if stage == 'autoencoding':
    dataset_file = args.dataset[0]
    split_percent = args.split_percent[0]
    encoding_dim_percent = args.encoding_dim_percent[0]
    output_file = args.differences_output_file[0]
    use_dbscan = args.use_dbscan

    autoencoding(dataset_file, split_percent, encoding_dim_percent, output_file, full_differences=use_dbscan)

elif stage == 'anomaly_selection':
    differences_file = args.differences_file[0]
    files_map_file = args.files_map_file[0]
    anomalies_output_file = args.anomalies_output_file[0]
    use_dbscan = args.use_dbscan

    anomalies_number =\
        anomaly_selection(files_map_file, anomalies_output_file, use_dbscan, differences_file=differences_file)

    print('===================' + os.linesep)
    print('%d anomalies found' % anomalies_number)

else:
    dataset_file = args.dataset[0]
    split_percent = args.split_percent[0]
    encoding_dim_percent = args.encoding_dim_percent[0]
    files_map_file = args.files_map_file[0]
    anomalies_output_file = args.anomalies_output_file[0]
    use_dbscan = args.use_dbscan

    differences = autoencoding(dataset_file, split_percent, encoding_dim_percent, full_differences=use_dbscan)

    anomalies_number = anomaly_selection(files_map_file, anomalies_output_file, use_dbscan, differences=differences)

    print('===================' + os.linesep)
    print('%d anomalies found' % anomalies_number)
