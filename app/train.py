import sys
from Utils.preprocess.preprocess import preprocess_data
import config
import traceback
import pandas as pd
from Utils.model_builder import RNN_pretrained_embed
from Utils.utlis import read_json_file
import os

hyper_param_path = config.HYPER_PARAM_PATH
data_schema = config.DATA_SCHEMA
data_path = config.DATA_PATH
failure_path = config.FAILURE_PATH 
save_model_path = config.MODEL_SAVE_PATH

def train():    
    try:        
        print('---------------------Training Started---------------------.')
        # Container reads training data from opt/ml_vol/inputs/data/training directory in mounted drive.
        data = pd.read_csv(data_path)
        # Container reads data config (schema) JSON file from the opt/ml_vol/inputs/data_config/ directory in mounted drive.
        preprocessor = preprocess_data(data=data) # no need to specify where the data schema, it reads it.
        # Container reads hyperparameters.json file from opt/ml_vol/model/model_config/ directory in mounted drive.
        hyperparametrs = read_json_file(hyper_param_path)

        # Container uses the three inputs above to train the preprocessor and the model and saves the artifacts in the opt/ml_vol/model/artifacts/ directory.
        model_trainer = RNN_pretrained_embed()
        preprocessor.drop_ids()
        x_train,y_train,x_val,y_val = preprocessor.get_train_test_data()

        model_trainer.fit(x_train=x_train,y_train=y_train,x_val=x_val,y_val=y_val,training_params=hyperparametrs)

        model_trainer.save_model(save_model_path)
        # If the task fails for any reason in any of the above steps, then container writes the failure reason in a file called train_failure in the
        
        # opt/ml_vol/outputs/errors/ directory.        



    except Exception as e:
        print("error!")
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        failure_file_path = os.path.join(failure_path,"train_failure.txt")
        with open(failure_file_path, 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == '__main__':
    train()