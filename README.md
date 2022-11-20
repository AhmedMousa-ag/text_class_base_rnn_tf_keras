# text_classification_tensorflow_bidirectional_RNN_model

## Navigatie Code

### app:

Is where all of code used to run train, test or serve modules.

config.py: is a file to determine main configurations such as files paths used during preprocessing, training, serving.

inference_app.py: is a file that decalere infer/ping and predictions to requested data happens

##### Utils:

Utils consist of:

1- preprocess folder: has the preprocess classes to process data according to schema.

2- model_builder.py: where the Machine Learning model defined, built, and loaded.

3- predictions_handler.py: called when needed a prediction for inference or testing.

4- utils.py: general functions to help such as load json files.

## Model architecture

Model architecture can be defined using hyperparameters.json file located at app/opt/ml_vol/model/model_config.

model consist of GRU RNN layer wrapped with bidirictional layer, number of layers can be defined in hyperparameters.json file.

{

"epochs":10, #Defines for how long to train the model.

"num_layers":2, #Defines number of Bidirectional Layers.

"neurons_num":50, #Defines number of neurons for each layer.

"embed_lay_output":120, #Defines the output dimension of the embedding layer "Not trained".

"learning_rate":0.01 #Defines the learning rate passed to Adam optimizer.

}

Each of parameters must be passed to build the model.
