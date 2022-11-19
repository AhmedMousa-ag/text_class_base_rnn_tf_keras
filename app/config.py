import os
from Utils.utlis import read_json_file


prefix = os.path.join('opt','ml_vol')

PREPROCESS_ARTIFACT_PATH = os.path.join("Utils","preprocess","artifacts")
if not os.path.exists(PREPROCESS_ARTIFACT_PATH):
    os.makedirs(PREPROCESS_ARTIFACT_PATH)

DATA_SCHEMA_PATH = os.path.join(prefix,"inputs","data_config","data_config_file.json")

TEXT_VECTORIZER_NAME = 'text_vectorizer.h5'

DATA_SCHEMA = read_json_file(DATA_SCHEMA_PATH)

FAILURE_PATH = os.path.join(prefix,'outputs','errors') 
if not os.path.exists(FAILURE_PATH):
    os.makedirs(FAILURE_PATH)

HYPER_PARAM_PATH = os.path.join(prefix,'model','model_config',"hyperparameters.json")


DATA_PATH = os.path.join(prefix,'inputs','data')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

TRAIN_DATA_PATH = os.path.join(DATA_PATH,'training',"train_data_file.csv") 
if not os.path.exists(TRAIN_DATA_PATH):
    os.makedirs(TRAIN_DATA_PATH)

TEST_DATA_PATH = os.path.join(DATA_PATH,'testing',"test_data_file.csv")
if not os.path.exists(TEST_DATA_PATH):
    os.makedirs(TEST_DATA_PATH)

MODEL_NAME= "tf_RNN_pretrained_embed"

MODEL_SAVE_PATH = os.path.join(prefix,'model','artifacts') 
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

SAVED_TEST_PRED_PATH = os.path.join(prefix,"outputs","testing_outputs")
if not os.path.exists(SAVED_TEST_PRED_PATH):
    os.makedirs(SAVED_TEST_PRED_PATH)


