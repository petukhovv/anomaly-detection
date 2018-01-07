import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', nargs=1, type=str, help='path to file')
parser.add_argument('--files_map_file', '-m', nargs=1, type=str,
                    help='file with map dataset indexes and ast file paths')
args = parser.parse_args()

path = args.path[0]
files_map_file = args.files_map_file[0]

with open(files_map_file) as f:
    files = json.loads(f.read())
    try:
        sample_number = files.index(path)
        print('Sample number: %d' % sample_number)
    except ValueError:
        print('File not found')
