# Automatizing-CNN-training-parameter-comparison-using-MLflow
A tool that was made to help deal with the combinatorial explosion problem faced while trying multiple combinations of hyperparameters. It uses MLflow to log all metrics generated by training and validating a neural network model using different hyperparameter combinations provided by the user. The main idea is to allow users to run 10s of models sequentially without the need for human intervention between runs thus elemenating any downtime.

## How to use:
### 1. Installation:
Anaconda is recommended to be used to run this tool. Creating a Virtual Environment named ${conda_env_name} is recommended. The following commands can be used to create a conda environment and install the required packages:
```
conda create -n ${conda_env_name} -y
conda activate ${conda_env_name}
conda install pip -y
pip install Keras
pip install Tensorflow
pip install Mlflow
```

### 2. Understand the runner.py file
runner.py acts as a tutorial on how to use the tool. It is recommended to read it first to get a feel of how the tool works.

### 3. Set custom input parameters
User imput is used to set the models and their hyperparameters to be trained and tested. The user has multiple options to give input to the tool.

#### Three types of inputs are possible:
1. Json file
2. Dictionary
3. Object of custom class (This also supports providing a custom model to train/test)

#### Json file:
The json file should be in the following format:
```
{
      "model_type": "vgg",
      "number_of_epocs": "10 20 30",
      "learning_rate": "0.001 0.005",
      "batch_size": "32",
      "loss": "categorical_crossentropy",
      "metrics": "accuracy",
      "optimizer": "adam sgd"
}
```
In this case the tool will train and test 12 models (3 epochs * 2 learning rates * 1 batch size * 1 loss * 1 metrics * 2 optimizers). Storing all outputs in MLflow.

#### Dictionary:
The dictionary follows the same format as the JSON file.

#### Custom class:
The ModelHyperparameters class should be in the following format:
```
Task = ModelHyperparameters('ResNet', 10, 32, 0.01, 'adam', 'categorical_crossentropy', 'accuracy')
```
where the constructor parameters are:
String / Keras model, number of epocs, batch size, learning rate, optimizer, loss, metrics
This method only supports training and testing one model at a time.

#### Three neural network models are baked into the tool:
These three models can be used by setting the model_type parameter to one of the following:
1. ResNet
2. VGG
3. AlexNet

### 4. Run runner.py
Running runner.py will train and test all models specified in all input methods. A file called mlruns will be generated to store models, performance metrics and hyperparameters using MLflow

### Output:
After each model is trained and tested the tool will log the results in MLflow. To view the MLflow logs open a terminal in ${Project_Path} and run the following commands:
```
conda activate ${conda_env_name}
mlflow ui
```
Then copy the URL given into your browser of choice. (The default URL is http://localhost:5000)

## Credits
A part of an academic project done under the supervision of Martin Dyrba

### Team members:
Hazem Ibrahim  
Igla Kalaja  
Mahla Haghzad  
Nafees Mohammad Adil  
Yasaman Ghassemipanah
