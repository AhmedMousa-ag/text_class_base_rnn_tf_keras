import sys
import Utils
from Utils.preprocess.preprocess import preprocess_data
import config
import traceback

hyper_param_path = config.HYPER_PARAM_PATH
data_schema_path = config.DATA_SCHEMA_PATH
data_path = config.DATA_PATH
artifact_path = config.PREPROCESS_ARTIFACT_PATH

def train():    
    try:        
        print('---------------------Training Started---------------------.')
        # Read in any hyperparameters that the user defined with algorithm submission
        hyper_parameters = Utils.get_hyperparameters(hyper_param_path)
        # Read data
        train_data = Utils.get_data(data_path)   
        # read data config
        data_schema = Utils.preprocess.schema_handler.get_data_schema(data_schema_path)
        # get trained preprocessor, model, training history 
        preprocessor, model = model_trainer.get_trained_model(train_data, data_schema, hyper_parameters)        
        # Save the processing pipeline   
        pipeline.save_preprocessor(preprocessor, model_path)
        # Save the model 
        classifier.save_model(model, model_path)    
        print('Done training.')
    except Exception as e:
        print("error!")
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        with open(failure_path, 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == '__main__':
    train()