import sys
import config
import traceback
import pandas as pd
from Utils.predictions_handler import Predictor
import os 
test_data_path = config.TEST_DATA_PATH
failure_path = config.FAILURE_PATH

def test():    
    try:        
        print('Starting test predictions')
    
        test_data = pd.read_csv(test_data_path)

        predictor = Predictor(test_data)

        predictor.save_predictions()


        print('Done test predictions.')
    except Exception as e:
        print("error!")
        # Write out an error file. This will be returned as the failureReason to the client.
        failure_file_path = os.path.join(failure_path,"test_failure.txt")
        trc = traceback.format_exc()
        with open(failure_file_path, 'w') as s:
            s.write('Exception during testing: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during testing: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)


if __name__ == '__main__':
    test()