import keras
from keras.layers.core import Dense, Dropout
from keras.models import Sequential, load_model, save_model
import tensorflow as tf
from scipy.sparse.csr import csr_matrix
import numpy as np

def read_model(model, filename):
    save_model(model, filename)

def write_model(filename):
    return load_model(filename)

class NeuralClassifier:
    @staticmethod
    def create_model(input_size, loss_name, num_classes):
        model = Sequential()

        model.add(Dense(15, input_dim=input_size, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss=loss_name,
                      metrics=['accuracy'])

        return model

    def __init__(self, X, loss_name, num_classes):
        self.model = NeuralClassifier.create_model(X[0].shape[1], loss_name, num_classes)

def update_data(X, Y):
    X = csr_matrix(X)

    num_classes = len(set(Y))

    start_num = min(list(set(Y)))
    y_train_new = []

    if start_num != 0:
        for i in range(len(Y)):
            Y[i] -= start_num

    loss_name = 'categorical_crossentropy'

    for i in Y:
        y_train_new.append(np.array([i]))

    y_train_new = np.array(y_train_new)

    one_hot_labels = keras.utils.to_categorical(y_train_new, num_classes=num_classes)
    Y = one_hot_labels

    X = X.todense()
    Y = np.array(Y)

    return X, Y, loss_name, num_classes

def train(X, Y):
    X, Y, loss_name, num_classes = update_data(X, Y)

    sess = tf.InteractiveSession()

    neural_net_class = NeuralClassifier(X, loss_name, num_classes)

    neural_net_class.model.fit(X, Y, epochs=30, batch_size=5)

    return neural_net_class.model


def test(model, X, Y):
    metrics = {}
    plots = {}

    return metrics, plots


def classify(model, features_arr):
    res_arr_proba = model.predict(features_arr)

    res_arr_class = []

    for i in res_arr_proba:
        max_index = i.tolist().index(max(i))
        res_arr_class.append(max_index)

    return res_arr_class, res_arr_proba
