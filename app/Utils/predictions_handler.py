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
        preds = tf.squeeze(preds)
        print(f"shape: {preds.shape}, and len is: {len(preds.shape)}")
        if len(preds.shape) == 1:
            return np.array(tf.round(preds), dtype=int)
        else:
            return np.array(tf.argmax(preds, axis=1), dtype=int)

    def save_predictions(self, save_path=SAVED_TEST_PRED_PATH):
        path = os.path.join(save_path, "test_predictions.csv")
        test_result = self.predict_get_results()
        test_result.to_csv(path)
        print(f"saved results to: {path}")
