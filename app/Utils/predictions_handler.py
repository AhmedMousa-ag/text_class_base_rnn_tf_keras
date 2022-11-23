import warnings
import numpy as np
import tensorflow as tf
import os
import config
import pandas as pd
from Utils.model_builder import load_model
from Utils.preprocess.preprocess import preprocess_data
import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

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

    def predict_test(self, data=None):  # called for test prediction
        if not data is None:
            self.preprocessor = preprocess_data(
                data, train=False, shuffle_data=False)

        ids = self.preprocessor.get_ids()
        self.preprocessor.drop_ids()

        processed_data = self.preprocessor.get_data()

        preds = self.model.predict(processed_data)

        preds_no_prob = self.conv_labels_no_probability(preds)

        num_uniq_preds = len(np.unique(preds_no_prob))
        print("number of unique preds: ",num_uniq_preds)
        uniqe_preds_names = np.squeeze(self.preprocessor.invers_labels(sorted(range(num_uniq_preds))))

        results_pd = pd.DataFrame([])
        results_pd['idField'] = ids

        if num_uniq_preds > 2:
            for i,uniqe in enumerate(uniqe_preds_names):
                results_pd[uniqe] = np.round(preds[:,i],5)
        else:
            #This means it's either 0 or 1
                pred = np.squeeze(preds)
                first_col_pred = []
                sec_col_pred = []
                for pred in np.squeeze(preds):
                    if pred > 0.5:
                        first_col_pred.append(pred-1)
                        sec_col_pred.append(pred)
                    else:
                        first_col_pred.append(pred)
                        sec_col_pred.append(pred-1)

                results_pd[uniqe_preds_names[0]] = np.round(first_col_pred,5)
                results_pd[uniqe_preds_names[1]] = np.round(sec_col_pred,5)



        # will convert get final prediction 
        preds = self.conv_labels_no_probability(preds)
        preds = self.preprocessor.invers_labels(preds)
        results_pd["prediction"] = preds
        results_pd = results_pd.sort_values(by=["idField"])
        return results_pd

    def predict_get_results(self, data=None):
        if not data is None:
            self.preprocessor = preprocess_data(
                data, train=False, shuffle_data=False)

        ids = self.preprocessor.get_ids()
        self.preprocessor.drop_ids()

        processed_data = self.preprocessor.get_data()

        preds = self.model.predict(processed_data)
        preds = self.conv_labels_no_probability(preds)

        preds = self.preprocessor.invers_labels(preds)

        results_pd = pd.DataFrame([])
        results_pd['idField'] = ids
        results_pd["prediction"] = preds
        results_pd = results_pd.sort_values(by=["idField"])
        return results_pd

    def conv_labels_no_probability(self, preds):
        preds = np.array(tf.squeeze(preds))
        if len(preds.shape) < 2:

            if preds.size < 2:  # If passed one prediction it cause and error if not expanded dimention
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
        # TODO replace it with the new function for predict_test
        test_result = self.predict_test()
        test_result.to_csv(path)
        print(f"saved results to: {path}")
