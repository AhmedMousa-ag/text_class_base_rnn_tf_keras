import os
import json




def read_json_file(file_path): 
    try:
        json_data = json.load(open(file_path)) 
        return json_data
    except: 
        raise Exception(f"Error reading json file at: {file_path}")   


def get_hyperparameters(hyper_param_path): 
    hyperparameters_path = os.path.join(hyper_param_path, 'hyperparameters.json')
    return read_json_file(hyperparameters_path, "hyperparameters")


def get_model_config():
    model_cfg_path = os.path.join(os.path.dirname(__file__), 'config', 'model_config.json')
    return read_json_file(model_cfg_path, "model config")

