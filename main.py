import math
import argparse
import json

from lib.Autoencoder import Autoencoder
from lib.DatasetLoader import DatasetLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-f', nargs=1, type=str, help='dataset file (csv format with colon delimiter)')
parser.add_argument('--split_percent', nargs=1, type=float, help='dataset train/test split percent')
parser.add_argument('--encoding_dim_percent', nargs=1, type=float,
                    help='encoding dim percent (towards features number)')
parser.add_argument('--output_file', '-o', nargs=1, type=str,
                    help='file with decoding losses (difference between input and output)')

args = parser.parse_args()

dataset_file = args.dataset[0]
split_percent = args.split_percent[0]
encoding_dim_percent = args.encoding_dim_percent[0]
output_file = args.output_file[0]

data = DatasetLoader(dataset_file).load(split_percent=split_percent)
(_, _, _, features_number) = data

encoding_dim = math.ceil(features_number * encoding_dim_percent)

autoencoder = Autoencoder(features_number, encoding_dim, data)
autoencoder.print_model_summary()
autoencoder.fit()
predicted = autoencoder.predict()

differences = autoencoder.calc_decoding_losses()
with open(output_file, 'w') as f:
    f.write(json.dumps(differences))
