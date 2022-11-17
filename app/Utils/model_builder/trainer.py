import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Bidirectional, GRU
from tensorflow.keras.metrics import Recall, Accuracy, Precision
import numpy as np
import config

MODEL_NAME = config.MODEL_NAME
MODEL_SAVE_PATH = config.MODEL_SAVE_PATH


class RNN_pretrained_embed():
    def __init__(self, neurons_num: int = 50, embed_lay_output: int = 120):
        self.neurons_num = neurons_num
        self.embed_lay_output = embed_lay_output

    def __build_model_compile(self, num_y_classes):
        """This fucntion builds the model and compile it"""
        # tensorflow hub universal sentence encoder
        optimizer = tf.keras.optimizers.Adam()
        metrics = [Precision(),
                   Recall(),
                   Accuracy()]

        embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                               trainable=False,
                               output_shape=self.embed_lay_output)

        model = tf.keras.Sequential()
        model.add(embed)
        model.add(Bidirectional(GRU(self.neurons_num)))
        if num_y_classes > 2:
            model.add(Dense(num_y_classes, activation='softmax'))
            loss = tf.keras.losses.CategoricalCrossentropy()
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        else:
            model.add(Dense(1, activation='binary_crossentropy'))
            loss = tf.keras.losses.BinaryCrossentropy()
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def fit(self, x_train, y_train, epochs, x_val=None, y_val=None):
        num_classes = len(np.unique(y_train))
        self.model = self.__build_model_compile(num_y_classes=num_classes)

        self.model.fit(x_train, y_train, validation_data=(
            x_val, y_val), validation_steps=len(x_val))

        return self.model


    def save_model(self,save_path=MODEL_SAVE_PATH):
        path = os.path.join(save_path,MODEL_NAME)
        self.model.save(path)

    @classmethod
    def load_model(self,save_path=MODEL_SAVE_PATH):
        path = os.path.join(save_path,MODEL_NAME)
        self.model = tf.load_model(path)
