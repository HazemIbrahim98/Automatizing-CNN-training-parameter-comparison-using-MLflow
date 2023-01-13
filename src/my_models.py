from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D, ZeroPadding2D, Input, Add, AveragePooling2D
from keras.models import Model
import keras.layers as layers
import keras.optimizers


class MyModels:
    def create(self, model, opt, lr, loss, metrics, input_shape, num_classes):
        if (type(model) is str):
            if model.lower() == 'vgg':
                return self.vgg(opt, lr, loss, metrics, input_shape, num_classes)
            elif model.lower() == 'alexnet':
                return self.alexNet(opt, lr, loss, metrics, input_shape, num_classes)
            elif model.lower() == 'resnet':
                return self.resNet(opt, lr, loss, metrics, input_shape, num_classes)
            # To add a new model, add a new elif statement here and a new function below
            else:
                print('Could not find model. Using vgg instead.')
                return self.vgg(opt, lr, loss, metrics, input_shape, num_classes)

        elif (type(model) is keras.engine.sequential.Sequential):
            model.compile(optimizer=self.getOptimizer(opt, lr), loss=self.getLoss(
                loss), metrics=self.getMetrics(metrics))
            return model
        else:
            print('Could not find model. Using vgg instead.')
            return self.vgg(opt, lr, loss, metrics, input_shape, num_classes)

    def vgg(self, opt, lr, loss, metrics, input_shape, num_classes):
        # From https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

        model = Sequential()

        model.add(Conv2D(input_shape=input_shape, filters=64,
                  kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3),
                  padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=256, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=num_classes, activation="softmax"))

        model.compile(optimizer=self.getOptimizer(opt, lr),
                      loss=self.getLoss(loss), metrics=self.getMetrics(metrics))
        return model

    def alexNet(self, opt, lr, loss, metrics, input_shape, num_classes):
        # From https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py

        model = Sequential()

        model.add(Conv2D(96, (11, 11), input_shape=input_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(3072))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        model.compile(optimizer=self.getOptimizer(opt, lr),
                      loss=self.getLoss(loss), metrics=self.getMetrics(metrics))
        return model

    def resNet(self, opt, lr, loss, metrics, input_shape, num_classes):
        # A simpliefied version of https://www.kaggle.com/code/akumaldo/resnet-from-scratch-keras/notebook

        X_input = Input(input_shape)
        X = ZeroPadding2D((3, 3))(X_input)

        X = Conv2D(32, (7, 7), strides=(1, 1))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3))(X)

        X = self.convolutional_block(
            X, f=3, filters=[32, 32, 128], stage=2, block='a', s=1)
        X = self.identity_block(X, 3, [32, 32, 128], stage=2, block='b')
        X = self.identity_block(X, 3, [32, 32, 128], stage=2, block='c')

        X = self.convolutional_block(
            X, f=3, filters=[64, 64, 256], stage=3, block='a', s=2)
        X = self.identity_block(X, 3, [64, 64, 256], stage=3, block='b')
        X = self.identity_block(X, 3, [64, 64, 256], stage=3, block='c')
        X = self.identity_block(X, 3, [64, 64, 256], stage=3, block='d')

        X = self.convolutional_block(
            X, f=3, filters=[128, 128, 512], stage=4, block='a', s=2)
        X = self.identity_block(X, 3, [128, 128, 512], stage=4, block='b')
        X = self.identity_block(X, 3, [128, 128, 512], stage=4, block='c')
        X = self.identity_block(X, 3, [128, 128, 512], stage=4, block='d')
        X = self.identity_block(X, 3, [128, 128, 512], stage=4, block='e')
        X = self.identity_block(X, 3, [128, 128, 512], stage=4, block='f')

        X = self.convolutional_block(
            X, f=3, filters=[256, 256, 1024], stage=5, block='a', s=2)
        X = self.identity_block(X, 3, [256, 256, 1024], stage=5, block='b')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=5, block='c')

        X = AveragePooling2D(pool_size=(2, 2))(X)

        X = Flatten()(X)
        X = Dense(num_classes, activation='softmax')(X)

        model = Model(inputs=X_input, outputs=X)

        model.compile(optimizer=self.getOptimizer(opt, lr),
                      loss=self.getLoss(loss), metrics=self.getMetrics(metrics))
        return model

    def identity_block(self, X, f, filters, stage, block):
        F1, F2, F3 = filters
        X_shortcut = X

        X = Conv2D(filters=F1, kernel_size=(1, 1),
                   strides=(1, 1), padding='valid')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F2, kernel_size=(f, f),
                   strides=(1, 1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F3, kernel_size=(1, 1),
                   strides=(1, 1), padding='valid')(X)
        X = BatchNormalization(axis=3)(X)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def convolutional_block(self, X, f, filters, stage, block, s=2):
        F1, F2, F3 = filters
        X_shortcut = X

        X = Conv2D(F1, (1, 1), strides=(s, s), padding='valid')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = Conv2D(F2, (f, f), strides=(1, 1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid')(X)
        X = BatchNormalization(axis=3)(X)

        X_shortcut = Conv2D(F3, (1, 1), strides=(s, s),
                            padding='valid')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def getOptimizer(self, opt, lr):
        if opt == 'adam':
            return keras.optimizers.Adam(learning_rate=lr)
        elif opt == 'sgd':
            return keras.optimizers.SGD(learning_rate=lr)
        elif opt == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=lr)
        elif opt == 'adagrad':
            return keras.optimizers.Adagrad(learning_rate=lr)
        elif opt == 'adadelta':
            return keras.optimizers.Adadelta(learning_rate=lr)
        elif opt == 'adamax':
            return keras.optimizers.Adamax(learning_rate=lr)
        elif opt == 'nadam':
            return keras.optimizers.Nadam(learning_rate=lr)
        else:
            print('Could not find optimizer. Using Adam instead.')
            return keras.optimizers.Adam(learning_rate=lr)

    def getLoss(self, loss):
        if loss == 'categorical_crossentropy':
            return keras.losses.CategoricalCrossentropy()
        elif loss == 'sparse_categorical_crossentropy':
            return keras.losses.SparseCategoricalCrossentropy()
        elif loss == 'mean_squared_error':
            return keras.losses.MeanSquaredError()
        elif loss == 'mean_absolute_error':
            return keras.losses.MeanAbsoluteError()
        elif loss == 'mean_absolute_percentage_error':
            return keras.losses.MeanAbsolutePercentageError()
        elif loss == 'mean_squared_logarithmic_error':
            return keras.losses.MeanSquaredLogarithmicError()
        elif loss == 'squared_hinge':
            return keras.losses.SquaredHinge()
        elif loss == 'hinge':
            return keras.losses.Hinge()
        elif loss == 'categorical_hinge':
            return keras.losses.CategoricalHinge()
        elif loss == 'logcosh':
            return keras.losses.LogCosh()
        elif loss == 'kullback_leibler_divergence':
            return keras.losses.KLDivergence()
        elif loss == 'poisson':
            return keras.losses.Poisson()
        elif loss == 'cosine_similarity':
            return keras.losses.CosineSimilarity()
        else:
            print('Could not find loss function. Using categorical_crossentropy instead.')
            return keras.losses.CategoricalCrossentropy()

    def getMetrics(self, metrics):
        return metrics.split(' ')
