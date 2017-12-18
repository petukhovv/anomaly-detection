from lib.Autoencoder import Autoencoder
from lib.DatasetLoader import DatasetLoader

(x_train, x_test, features_number) = DatasetLoader('dataset.csv').load(split_percent=0.1)

autoencoder = Autoencoder(features_number=features_number, encoding_dim=1000)
autoencoder.print_model_summary()
autoencoder.fit(x_train, x_test)

losses = autoencoder.get_losses()

print(losses)
