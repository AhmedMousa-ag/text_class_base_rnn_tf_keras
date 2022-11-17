from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import pickle
import os
import config
from Utils.preprocess.schema_handler import produce_schema_param
import pandas as pd
import numpy as np

ARTIFACTS_PATH = config.PREPROCESS_ARTIFACT_PATH
DATA_SCHEMA = config.DATA_SCHEMA

class preprocess_data():
    def __init__(self, data, data_schema=DATA_SCHEMA,artifacts_path=ARTIFACTS_PATH, shuffle_data=True, train=True):
        """
        args:
            data: The data we want to preprocess
            data_schema: The schema that will handle the data
            shuffle_data: If True it will shuffle the data before processing it
            artifacts_path: The path to any saved/will save preprocess tool such as LabelEncoder
            train: if it's True it will save artifacts to use later in serving or testing
        """
        if not isinstance(data, pd.DataFrame):  # This should handle if the passed data is json or something else
            self.data = pd.DataFrame(data)
        else:
            self.data = data

        self.data_schema = data_schema
        self.schema_param = produce_schema_param(self.data_schema)
        self.artifacts_path = artifacts_path
        self.train = train
        self.LABELS = self.define_labels() # Get's labels columns

        self.clean_data() # Checks for dublicates or null values and removes them

        if shuffle_data:
            self.data.sample(frac=1).reset_index(drop=True)

        self.fit_transform() # preprocess data based on the schema

    def clean_data(self):
        if self.data.duplicated().sum() > 0:
            self.data.drop_duplicates(inplace=True)
        
        if self.data.isnull().sum() > 0:
            self.data.dropna(inplace=True)

        self.data.reset_index(drop=True)


    def fit_transform(self):
        ''' preprocess data based on the schema, in case it's not training then it will load the preprocess pickle object'''
        for key in self.schema_param.keys():
            if key == "id":
                # It does nothing, but in case we decided to do something in the future
                self.data[key] = prep_NUMERIC.handle_id(self.data[key])
            elif key == "class":  # Will assume it's label and startes to label encode it
                self.data[key] = prep_NUMERIC.LabelEncoder(
                    self.data[key], key, self.artifacts_path, self.train)
            elif key == "txt":
                self.data[key] = prep_TEXT.get_process_text(
                    self.data[key], key, self.artifacts_path, self.train)

    def define_labels(self):
        labels = []
        for key in self.schema_param.keys:
            if "target" in key:
                labels.append(key)

        if len(labels) == 1:  # If it's one labels then will return a string of that label only
            return labels[0]
        else:   # Otherwise it returns a list of labels
            return labels

    def drop_ids(self):
        self.data.drop('idField',axis=1,inplace=True)

    def get_ids(self):
        return self.data['idField']

    def __split_x_y(self):
        self.y_data = self.data[self.LABEL]
        self.x_data = self.data.drop([self.LABEL], axis=1)
        return self.x_data, self.y_data

    def __train_test_split(self, train_ratio=0.8):
        self.__split_x_y()
        x_train_indx = int(train_ratio*len(self.x_data))
        self.x_train = self.x_data.iloc[:x_train_indx, :]

        if isinstance(self.LABEL, str):  # If it's one single label not multiple labels
            self.y_train = self.y_data.iloc[:x_train_indx]
            self.y_test = self.y_data.iloc[x_train_indx:]
        else:  # If it's multiple labels
            self.y_train = self.y_data.iloc[:x_train_indx, :]
            self.y_test = self.y_data.iloc[x_train_indx:, :]

        self.x_test = self.x_data.iloc[x_train_indx:, :]

        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_train_test_data(self):
        """returns: 
            x_train, y_train, x_test, y_test
        """
        self.__train_test_split()
        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_data(self):
        return self.data

# ----------------------------------------------------------
class prep_TEXT():
    def __init__(self):
        pass

    def get_process_text(self, data, col_name=None, artifacts_path=None, Training=False):
        """Univeral encoder handles it so will just return it as it's"""
        return data

# -----------------------------------------------------------
class prep_NUMERIC():
    def __init__(self):
        pass

    @classmethod
    def LabelEncoder(self, data, col_name, artifacts_path, Training=False):
        path = os.path.join(artifacts_path, col_name+".pkl")
        if Training:
            encoder = LabelEncoder()
            encoded_data = encoder.fit_transform(data)
            pickle.dump(encoder, open(path, 'wb'))
        else:
            encoder = pickle.loads(path)
            encoded_data = encoder.transform(data)
        return encoded_data

    @classmethod
    def handle_id(self, data):
        return data

    @classmethod
    def Min_Max_Scale(self, data, col_name, artifacts_path, Training=False):
        path = os.path.join(artifacts_path, col_name+".pkl")
        if self.Training:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
            pickle.dump(scaler, open(path, 'wb'))
        else:
            scaler = pickle.loads(path)
            scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
        return scaled_data

    @classmethod
    def Standard_Scale(self, data, col_name, artifacts_path, Training=False):
        path = os.path.join(artifacts_path, col_name+".pkl")
        if self.Training:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
            pickle.dump(scaler, open(path, 'wb'))
        else:
            scaler = pickle.loads(path)
            scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
        return scaled_data
