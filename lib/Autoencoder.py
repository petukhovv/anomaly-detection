from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import Callback


class AutoencoderModel:
    def __init__(self, features_number, encoding_dim):
        self.features_number = features_number
        self.encoding_dim = encoding_dim

        input_img = Input(shape=(features_number,))
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        decoded = Dense(features_number, activation='sigmoid')(encoded)

        self.autoencoder = Model(input_img, decoded)

        self.encoder = Model(input_img, encoded)

        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

    def compile(self, optimizer='adadelta', loss='binary_crossentropy'):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def summary(self):
        self.autoencoder.summary()

    def get_autoencoder(self):
        return self.autoencoder


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class Autoencoder:
    def __init__(self, features_number, encoding_dim):
        self.model = AutoencoderModel(features_number=features_number, encoding_dim=encoding_dim)
        self.model.compile()
        self.history = None

    def print_model_summary(self):
        self.model.summary()

    def fit(self, x_train, x_test, epochs=5, shuffle=True):
        self.history = LossHistory()
        self.model.get_autoencoder().fit(x_train, x_train,
                                         epochs=epochs,
                                         shuffle=shuffle,
                                         validation_data=(x_test, x_test),
                                         callbacks=[self.history])

    def get_losses(self):
        return self.history.losses
