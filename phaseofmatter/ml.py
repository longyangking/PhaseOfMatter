from __future__ import print_function

import numpy as np 
import os

import tensorflow as tf 
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras import regularizers
import keras.backend as K 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only error will be shown

class MLModel:
    def __init__(self,
        state_shape,
        verbose=False
    ):
        self.state_shape = state_shape
        self.learning_rate = 1e-3
        self.l2_const = 1e-4
        self.verbose = verbose

        self.model = self.build_model()
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(self.learning_rate),
            metrics=['acc']
        )

    def build_model(self):
        state_tensor = Input(shape=self.state_shape)

        out = Conv2D(filters = 64,
            kernel_size = (3,3),
            activation='relu',
            kernel_regularizer = regularizers.l2(self.l2_const)
            )(state_tensor)

        out = MaxPooling2D()(out)

        out = Flatten()(out)
        out = Dense(64, activation='relu', kernel_regularizer = regularizers.l2(self.l2_const))(out)
        output_tensor = Dense(1, activation='sigmoid')(out)

        return Model(inputs=state_tensor, outputs=output_tensor)

    def fit(self, Xs, ys, epochs, batch_size):
        history = self.model.fit(Xs, ys, epochs=epochs, batch_size=batch_size, verbose=self.verbose)
        return history

    def predict(self, Xs):
        outputs = self.model.predict(Xs)
        return outputs
    
    def evaluate(self, Xs, ys):
        return self.model.evaluate(Xs, ys)

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)

    def plot_model(self, filename):
        from keras.utils import plot_model
        plot_model(self.model, show_shapes=True, to_file=filename)
    