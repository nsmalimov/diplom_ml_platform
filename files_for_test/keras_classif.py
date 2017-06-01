import keras
from keras.models import Sequential
import tensorflow as tf
from scipy.sparse.csr import csr_matrix
import numpy as np

class NeuralClassifier():
    def neural_model_wine(X, y):
        X = csr_matrix(X)

        y = [int(i) for i in y]

        y_train_new = []

        for i in y:
            y_train_new.append(np.array([i]))

        y_train_new = np.array(y_train_new)

        y_train_new[y_train_new == 1] = 0
        y_train_new[y_train_new == 2] = 1
        y_train_new[y_train_new == 3] = 2

        one_hot_labels = keras.utils.to_categorical(y_train_new, num_classes=3)

        # tf.reset_default_graph()
        sess = tf.InteractiveSession()

        model = Sequential()

        # model.add(Dense(4, input_dim=X[0].shape[1], kernel_initializer='normal', activation='relu'))
        # model.add(Dense(24, kernel_initializer='normal', activation='sigmoid'))
        # model.add(Dense(12, kernel_initializer='normal', activation='sigmoid'))
        # model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
        #
        # model.compile(optimizer='adam',
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])

        model.add(Dense(15, input_dim=X[0].shape[1], kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.1))
        # model.add(Dense(12, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.1))
        model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("start fit")
        model.fit(X.todense(), one_hot_labels, epochs=30, batch_size=5)

        return model


def train(X, Y):
    Y_new = list(set(Y))
    num_classes = len(Y_new)
    start_num = min(Y_new)

    lr = RandomClassifier(num_classes, start_num)
    lr.fit(X, Y)

    return lr


def test(model, X, Y):
    metrics = {}
    plots = {}

    return metrics, plots


def classify(model, features_arr):
    res_arr_class = [model.predict(i) for i in features_arr]
    return res_arr_class, None
