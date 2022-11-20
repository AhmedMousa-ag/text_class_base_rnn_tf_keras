import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, GRU,Lambda, Flatten, Embedding, Input
from tensorflow.keras.metrics import Recall, Accuracy, Precision
import numpy as np
import config
from Utils.preprocess.preprocess import prep_TEXT

MODEL_NAME = config.MODEL_NAME
MODEL_SAVE_PATH = config.MODEL_SAVE_PATH


class RNN_pretrained_embed():
    def __init__(self):
        pass

    def __build_model_compile(self, num_y_classes, num_layers, neurons_num, embed_lay_output,learning_rate):
        """This fucntion builds the model and compile it"""
        # tensorflow hub universal sentence encoder
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        metrics = [Precision(),
                   Recall()
                   ]
        
        text_vectorizer = prep_TEXT.load_text_vectorizer()
        
        max_vocab_length = len(text_vectorizer.get_vocabulary())

        max_length = len(tf.squeeze(text_vectorizer(["dsads"]))) #We defined output lenght during preprocessing, now getting it for embedding layer

        embed_layer = Embedding(input_dim=max_vocab_length, # set input shape
                             output_dim=embed_lay_output, # set size of embedding vector
                             input_length=max_length, # how long is each input
                             name="Embedding_Layer") 

        model = tf.keras.Sequential()
        model.add(Input(shape=[1,],dtype=tf.string))
        model.add(text_vectorizer)
        model.add(embed_layer)

        for i in range(num_layers):
            model.add(Bidirectional(GRU(neurons_num,return_sequences=True),
                        name=f"Bidirectional_layer_{i}"))

        model.add(Flatten())

        if num_y_classes > 2:
            model.add(Dense(num_y_classes, activation='softmax'))
            loss = tf.keras.losses.CategoricalCrossentropy()
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        else:                             
            model.add(Dense(1, activation='sigmoid'))
            loss = tf.keras.losses.BinaryCrossentropy()
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def fit(self, x_train, y_train, training_params, x_val=None, y_val=None):
        num_classes = len(np.unique(y_train))

        epochs = training_params['epochs']
        num_layers = training_params['num_layers']
        neurons_num = training_params['neurons_num']
        embed_lay_output = training_params['embed_lay_output']
        lr = training_params['learning_rate']
        self.model = self.__build_model_compile(
            num_y_classes=num_classes, num_layers=num_layers,
            neurons_num=neurons_num, embed_lay_output=embed_lay_output,
            learning_rate=lr)
            
        if num_classes>2:
            y_train = tf.squeeze(tf.one_hot(y_train,num_classes))
            if not y_val is None:
                y_val = tf.squeeze(tf.one_hot(y_val,num_classes))

        if x_val is None:
            self.model.fit(x_train, y_train, epochs=epochs)
        else:
            
            self.model.fit(x_train, y_train, epochs=epochs, validation_data=(
                x_val, y_val), validation_steps=len(x_val))

        return self.model

    def save_model(self, save_path=MODEL_SAVE_PATH):
        path = os.path.join(save_path, MODEL_NAME)
        self.model.save(path)


def load_model(save_path=MODEL_SAVE_PATH):
    path = os.path.join(save_path, MODEL_NAME)
    model = tf.keras.models.load_model(path)
    print(f"Loaded model from: {path} successfully")
    return model
