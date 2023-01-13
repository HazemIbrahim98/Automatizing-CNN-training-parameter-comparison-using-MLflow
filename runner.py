from src.tool import tool
from src.model_hyperparameters import ModelHyperparameters

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
import keras.layers as layers

# Function that returns a Keras model.
# Only used to keep the main function simple.


def createModel():
    model = Sequential()
    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    return model


if (__name__ == '__main__'):
    # Create a tool object
    mytool = tool()

    # Set the experiment name for the MLFlow UI
    mytool.setExperimentName('Default')

    # Set the seed for reproducibility
    mytool.setSeed(0)

    # Schedule a dict
    '''
    # To schedule multiple dicts:
    dicts = [
        {
        "model_type": "VGG",
        ...
        },
        {
        "model_type": "AlexNet",
        ...
        }
    ]
    for i in dicts:
        mytool.schedule(i)
    '''
    thisdict = {
        "model_type": "resnet",
        "number_of_epocs": "5",
        "learning_rate": "0.001",
        "batch_size": "32",
        "loss": "categorical_crossentropy",
        "metrics": "accuracy mse",
        "optimizer": "adam"
    }
    mytool.schedule(thisdict)

    # Schedule a JSON file
    mytool.schedule('input.json')

    # Schedule a ModelHyperparameters object. Takes the same parameters like the JSON and dict.
    # Parameters: String / Keras model, number of epocs, batch size, learning rate, optimizer, loss, metrics
    task = ModelHyperparameters(
        'ResNet', 10, 32, 0.01, 'adam', 'categorical_crossentropy', 'accuracy')
    mytool.schedule(task)

    # Schedule a Custom model, Make a custom keras.engine.sequential.Sequential Model and pass it in a ModelHyperparameters object
    model = createModel()

    # Parameters: String / Keras model, number of epocs, batch size, learning rate, optimizer, loss, metrics
    mytool.schedule(ModelHyperparameters(model, 10, 32, 0.001,
                    'adam', 'categorical_crossentropy', 'accuracy'))

    # Run the tool
    mytool.run()
