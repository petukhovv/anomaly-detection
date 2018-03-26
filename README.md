# anomaly-detection

Anomaly detection using autoencoder and dbscan (adapted for Kotlin source code anomaly detection)

Available steps (stages):
- `autoencoding`: run the autoencoder on the specified dataset, calulating and write vectors with differences (between input and output) or simple euclidean distances array;
- `anomaly_selection`: read difference vectors or distances vector and anomaly selection in it (via DBScan or 5-sigma);
- `[without specify stage]`: run of both stages without intermediate write of difference vectors or distances vector in a file.

## Program use
### Autoencoding

#### Description

At this stage, the vectors representing the AST are encoded and decoded, and the differences between the input vectors and decoded vectors is written to a file

#### Stage arguments

* `-s`, `--stage` -> autoencoding;
* `--use_dbscan` (*default=false*): whether to use dbscan (**high memory or time usage!**) - then will use full differences between autoencoder input and output vectors matrix; if not, then will use simple euclidean distance between autoencoder input and output vectors;
* `-f`, `--dataset`: path to dataset file (csv format with colon delimiter);
* `--split_percent`: dataset train/test split percent;
* `--encoding_dim_percent`: encoding dim percent (towards features number);
* `--differences_output_file`: path to file with input-decoded difference (full differences matrix if --use_dbscan=True or simple distances vector if not).

#### Example of use

With DBScan:
```
python3 main.py -s autoencoding --use_dbscan -f dataset.csv --split_percent 0.9 --encoding_dim_percent 0.8 --differences_output_file differences.bin
```

Without DBScan:
```
python3 main.py -s autoencoding -f dataset.csv --split_percent 0.9 --encoding_dim_percent 0.8 --differences_output_file distances.json
```

If use full differences matrix (with --use_dbscan option), then the file will be written in binary mode.

### Anomaly selection

#### Description

At this stage, anomalies are selected by the difference matrix (via DBScan) or the distances vector (via 5-sigma)

#### Stage arguments

* `-s`, `--stage` -> autoencoding;
* `--use_dbscan` (*default=false*): whether to use dbscan (**high memory or time usage!**) - then will use full differences between autoencoder input and output vectors matrix; if not, then will use simple euclidean distance between autoencoder input and output vectors;
* `--differences_file`: path to file with input-decoded difference (full differences matrix or simple distances vector), obtained previous stage (autoencoding);
* `--files_map_file`: file with map dataset indexes and ast file paths, obtained by [ast-set2matrix](https://github.com/PetukhovVictor/ast-set2matrix) with stage=vectors2matrix;
* `-o`, `--anomalies_output_file`: path to file, which will contain ranking anomaly list (as paths to AST code snippets and ranks);

#### Example of use

With DBScan:
```
python3 main.py -s anomaly_selection --use_dbscan --differences_file differences.bin --files_map_file files_map.json --anomalies_output_file anomalies.json
```

Without DBScan:
```
python3 main.py -s anomaly_selection --differences_file distances.json --files_map_file files_map.json --anomalies_output_file anomalies.json
```

If use full differences matrix (with `--use_dbscan` option), then the file will be read in binary mode.

### Without specify stage

If you do not specify a stage, then runs both stages.

Use arguments both stages except `--differences_output_file` and `--differences_file`.

#### Example of use

With DBScan:
```
python3 main.py -f dataset.csv --files_map_file files_map.json --split_percent 0.9 --encoding_dim_percent 0.8 --anomalies_output_file anomalies.json
```

Without DBScan:
```
python3 main.py --use_dbscan -f dataset.csv --files_map_file files_map.json --split_percent 0.9 --encoding_dim_percent 0.8 --anomalies_output_file anomalies.json
```
