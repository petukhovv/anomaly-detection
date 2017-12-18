import math
import argparse

from lib.Autoencoder import Autoencoder
from lib.DatasetLoader import DatasetLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-f', nargs=1, type=str, help='dataset file (csv format with colon delimiter)')
parser.add_argument('--split_percent', nargs=1, type=float, help='dataset train/test split percent')
parser.add_argument('--encoding_dim_percent', '-o', nargs=1, type=float,
                    help='encoding dim percent (towards features number)')

args = parser.parse_args()

dataset_file = args.dataset[0]
split_percent = args.split_percent[0]
encoding_dim_percent = args.encoding_dim_percent[0]

(x_train, x_test, features_number) = DatasetLoader(dataset_file).load(split_percent=split_percent)

encoding_dim = math.ceil(features_number * encoding_dim_percent)
autoencoder = Autoencoder(features_number=features_number, encoding_dim=encoding_dim)
autoencoder.print_model_summary()
autoencoder.fit(x_train, x_test)

losses = autoencoder.get_losses()

print(losses)
