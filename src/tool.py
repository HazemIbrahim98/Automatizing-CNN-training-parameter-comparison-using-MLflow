from src.model_hyperparameters import ModelHyperparameters
from src.my_models import MyModels

import json
import mlflow

from tensorflow.keras import datasets
from keras.utils import to_categorical
import numpy as np

from random import randrange
import tensorflow
from numpy.random import seed


class tool:
    def __init__(self):
        self.tasks = []
        self.seed = randrange(1000)
        self.mlflowExperementName = 'Default'

    def setExperimentName(self, name):
        self.mlflowExperementName = name

    def setSeed(self, seed):
        if (type(seed) is int):
            self.seed = seed
            print('Setting seed to ', seed)
        else:
            print('Seed must be an integer')

    def schedule(self, input):
        if (type(input) is str):
            with open(input, 'r') as f:
                data = json.load(f)

            for i in data['Tasks']:
                self.decodeTasks(i)
            print('Scheduled a total of', len(self.tasks), 'models')
            return

        elif (type(input) is dict):
            self.decodeTasks(input)
            print('Scheduled a total of', len(self.tasks), 'models')
            return

        elif (type(input) is ModelHyperparameters):
            self.tasks.append(input)
            print('Scheduled a total of', len(self.tasks), 'models')
            return

    def decodeTasks(self, i):
        try:
            for epoc in i['number_of_epocs'].split(' '):
                for batch in i['batch_size'].split(' '):
                    for lr in i['learning_rate'].split(' '):
                        for loss in i['loss'].split(' '):
                            for opt in i['optimizer'].split(' '):
                                task = ModelHyperparameters(
                                    i['model_type'], epoc, batch, lr, opt, loss, i['metrics'])
                                self.tasks.append(task)
        except:
            print('Error decoding task, there is a problem with', i)
            return

    def getDataset(self):
        print('Getting dataset from the internet')

        (data_train, label_train), (data_test,
                                    label_test) = datasets.cifar10.load_data()
        data_train, data_test = data_train / 255.0, data_test / 255.0

        label_train = to_categorical(np.asarray(label_train))
        label_test = to_categorical(np.asarray(label_test))

        return data_train, label_train, data_test, label_test

    def returnModel(self, modelHyperParameters, input_shape, num_classes):
        myModels = MyModels()
        return myModels.create(modelHyperParameters.model_type, modelHyperParameters.optimizer, modelHyperParameters.learning_rate, modelHyperParameters.loss, modelHyperParameters.metrics, input_shape, num_classes)

    def run(self):
        if (len(self.tasks) == 0):
            print("No models to run, Please schedule models using schedule()")
            return

        data_train, label_train, data_test, label_test = self.getDataset()

        input_shape = data_train.shape[1:]
        num_classes = label_train.shape[1]

        mlflow.set_experiment(self.mlflowExperementName)
        experiment = mlflow.get_experiment_by_name(self.mlflowExperementName)

        for i in range(len(self.tasks)):
            try:
                with mlflow.start_run(experiment_id=experiment.experiment_id, nested=True, run_name=str(i + 1) + ' of ' + str(len(self.tasks))):
                    print('Running model number', i + 1, 'of', len(self.tasks))
                    mlflow.tensorflow.autolog()
                    mlflow.log_param("seed", self.seed)
                    mlflow.log_param("Loss", self.tasks[i].loss)
                    if (type(self.tasks[i].model_type) is str):
                        mlflow.log_param(
                            "Model", self.tasks[i].model_type.upper())
                    else:
                        mlflow.log_param("Model", "Custom model")
                    self.runModel(self.tasks[i], data_train, label_train,
                                  data_test, label_test, input_shape, num_classes)
                    mlflow.end_run()
            except:
                print('Error running model number', i + 1,
                      'of', len(self.tasks), 'skipping')
                continue
        self.tasks = []

    def runModel(self, modelHyperParameters, data_train, label_train, data_test, label_test, input_shape, num_classes):
        seed(self.seed)
        tensorflow.random.set_seed(self.seed)

        # Making model
        model = self.returnModel(modelHyperParameters,
                                 input_shape, num_classes)

        # Running model
        history = model.fit(data_train, label_train, epochs=modelHyperParameters.number_of_epocs,
                            batch_size=modelHyperParameters.batch_size, validation_data=(data_test, label_test), verbose=1)
