import os
from Utils.utlis import read_json_file


prefix = os.path.join('opt','ml_vol')

PREPROCESS_ARTIFACT_PATH = os.path.join("Utils","preprocess","artifacts")
if not os.path.exists(PREPROCESS_ARTIFACT_PATH):
    os.makedirs(PREPROCESS_ARTIFACT_PATH)

DATA_SCHEMA_PATH = os.path.join(prefix,"inputs","data_config","data_config_file.json")

DATA_SCHEMA = read_json_file(DATA_SCHEMA_PATH)

FAILURE_PATH = os.path.join(prefix,'outputs','errors') 

HYPER_PARAM_PATH = os.path.join(prefix,'model','model_config')

DATA_PATH = os.path.join(prefix,'inputs','data')

TRAIN_DATA_PATH = os.path.join(DATA_PATH,'training') 

TEST_DATA_PATH = os.path.join(DATA_PATH,'testing')

MODEL_NAME= "tf_RNN_pretrained_embed.h5"

MODEL_SAVE_PATH = os.path.join(prefix,'model','artifacts') 


SAVED_TEST_PRED_PATH = os.path.join(prefix,"outputs","testing_outputs")