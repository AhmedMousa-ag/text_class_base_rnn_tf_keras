from Utils.preprocess.preprocess import preprocess_data
from Utils.model_builder import load_model
import pandas as pd
import config
import os

SAVED_TEST_PRED_PATH = config.SAVED_TEST_PRED_PATH

class Predictor():
    def __init__(self,data):
        self.model = load_model()
        self.preprocessor = preprocess_data(data,train=False)

    def predict_get_results(self):
        ids = self.preprocessor.get_ids()
        self.preprocessor.drop_ids()

        processed_data = self.preprocessor.get_data()

        preds = self.model.predict(processed_data)

        results_pd = pd.DataFrame([])
        results_pd['idField'] = ids
        results_pd["prediction"] = preds
        results_pd = results_pd.sort_values(by=["idField"])
        return results_pd

    def save_predictions(self,save_path = SAVED_TEST_PRED_PATH):
        path = os.path.join(save_path,"test_predictions.csv")
        test_result = self.predict_get_results()
        test_result.save_csv(path)
        print(f"saved results to: {path}")