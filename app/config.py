import os
from Utils.utlis import read_json_file

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

prefix = os.path.join('opt','ml_vol')

PREPROCESS_ARTIFACT_PATH = os.path.join("Utils","preprocess","artifacts")
check_dir(PREPROCESS_ARTIFACT_PATH)


DATA_SCHEMA_PATH = os.path.join(prefix,"inputs","data_config","data_config_file.json")
check_dir(os.path.join(prefix,"inputs","data_config"))


TEXT_VECTORIZER_NAME = 'text_vectorizer.h5'

DATA_SCHEMA = read_json_file(DATA_SCHEMA_PATH)

FAILURE_PATH = os.path.join(prefix,'outputs','errors','serve_failure.txt') 
check_dir(FAILURE_PATH)


HYPER_PARAM_PATH = os.path.join(prefix,'model','model_config',"hyperparameters.json")
check_dir(os.path.join(prefix,'model','model_config'))


DATA_PATH = os.path.join(prefix,'inputs','data')
check_dir(DATA_PATH)
check_dir(os.path.join(DATA_PATH,"training"))
check_dir(os.path.join(DATA_PATH,"testing"))


TRAIN_DATA_PATH = os.path.join(DATA_PATH,'training',"train_data_file.csv") 


TEST_DATA_PATH = os.path.join(DATA_PATH,'testing',"test_data_file.csv")


MODEL_NAME= "tf_bidirectional_text_class"

MODEL_SAVE_PATH = os.path.join(prefix,'model','artifacts') 
check_dir(MODEL_SAVE_PATH)

SAVED_TEST_PRED_PATH = os.path.join(prefix,"outputs","testing_outputs")
check_dir(SAVED_TEST_PRED_PATH)

