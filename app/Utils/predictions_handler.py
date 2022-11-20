from Utils.preprocess.preprocess import preprocess_data
from Utils.model_builder import load_model
import pandas as pd
import config
import os
import tensorflow as tf
import numpy as np

SAVED_TEST_PRED_PATH = config.SAVED_TEST_PRED_PATH


class Predictor():
    def __init__(self, data=None, model=None):

        if model is None:
            self.model = load_model()
        else:  # Modelshould be reloaded before getting the reques, that's the reason to pass the model to the predictor
            self.model = model

        if not data is None:
            self.preprocessor = preprocess_data(
                data, train=False, shuffle_data=False)

    def predict_get_results(self, data=None):
        if not data is None:
            self.preprocessor = preprocess_data(
                data, train=False, shuffle_data=False)

        ids = self.preprocessor.get_ids()
        self.preprocessor.drop_ids()

        processed_data = self.preprocessor.get_data()

        preds = self.model.predict(processed_data)
        preds = self.conv_labels_no_probability(preds)

        print("preds are: ", preds)

        preds = self.preprocessor.invers_labels(preds)

        results_pd = pd.DataFrame([])
        results_pd['idField'] = ids
        results_pd["prediction"] = preds
        results_pd = results_pd.sort_values(by=["idField"])
        return results_pd

    def conv_labels_no_probability(self, preds):
        preds = np.array(tf.squeeze(preds))
        if len(preds.shape) < 2:

            if preds.size < 2:  # If passed one prediction it cause and error if not expanded dimenstion
                prediction = np.array(tf.expand_dims(
                    tf.round(preds), axis=0), dtype=int)
            else:
                prediction = np.array(tf.round(preds), dtype=int)

            return prediction
        else:

            if preds.size < 2:  # If passed one prediction it cause and error if not expanded dimenstion
                prediction = np.array(tf.expand_dims(
                    tf.argmax(preds, axis=1), axis=0), dtype=int)
            else:
                prediction = np.array(tf.argmax(preds, axis=1), dtype=int)

            return prediction

    def save_predictions(self, save_path=SAVED_TEST_PRED_PATH):
        path = os.path.join(save_path, "test_predictions.csv")
        test_result = self.predict_get_results()
        test_result.to_csv(path)
        print(f"saved results to: {path}")
