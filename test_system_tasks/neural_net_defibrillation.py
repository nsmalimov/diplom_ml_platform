import keras
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential, save_model, load_model
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from keras import metrics

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

X, Y = [], []

#f = open("/Users/Nurislam/PycharmProjects/diplom_project_2/data/concated/commonData.csv", "r")

path = "/home/nur/PycharmProjects/diplom_ml_platform/test/data/"

f = open(path + "commonData.csv", "r")

for i in f.readlines():
    s = i.replace("\n", "")
    arr = s.split(",")

    X.append([float(i) for i in arr[:-1]])
    Y.append(arr[-1])

f.close()

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

def proportional_split_test_train(X, Y):
    X_train, X_test, Y_train, Y_test = [], [], [], []

    split_num = 0.33
    each_elem_count = {}
    need_each_count = {}

    for i in Y:
        if not(i in each_elem_count):
            each_elem_count[i] = 1
            need_each_count[i] = 0
        else:
            each_elem_count[i] += 1

    for i in each_elem_count:
        need_each_count[i] = int(each_elem_count[i] * split_num)

    for i in range(len(X)):
        label = Y[i]
        elem = X[i]

        if need_each_count[label] >= 0:
            X_test.append(elem)
            Y_test.append(label)
            need_each_count[label] -= 1
        else:
            X_train.append(elem)
            Y_train.append(label)

    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = proportional_split_test_train(X, Y)

def neural_model(X, y):
    X = csr_matrix(X)

    y = [int(i) for i in y]

    y_train_new = []

    for i in y:
        y_train_new.append(np.array([i]))

    y_train_new = np.array(y_train_new)

    #y_train_new[y_train_new == 1] = 0
    #y_train_new[y_train_new == 2] = 1
    #y_train_new[y_train_new == 3] = 2

    one_hot_labels = keras.utils.to_categorical(y_train_new, num_classes=2)

    #tf.reset_default_graph()
    sess = tf.InteractiveSession()

    model = Sequential()

    model.add(Dense(4, input_dim=X[0].shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(24, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(12, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print ("start fit")
    model.fit(X.todense(), one_hot_labels, epochs=30, batch_size=5)

    return model

def model_neural_test(X_test, Y_test):
    model = load_model("/home/nur/PycharmProjects/diplom_ml_platform/test_system_tasks/neural_model_def")

    predict = model.predict(X_test)

    res_arr = []

    for i in predict:
        print (i)
        max_index = i.tolist().index(max(i))
        res_arr.append(max_index + 1)

    mean_score = 0

    for i in range(len(res_arr)):
        print (str(res_arr[i]) + " " + str(Y_test[i]))
        mean_score += 1 if res_arr[i] == int(Y_test[i]) else 0

    print (mean_score)

    print (mean_score / (len(res_arr) + 0.0))


model = neural_model(X_train, Y_train)

save_model(model, "/home/nur/PycharmProjects/diplom_ml_platform/test_system_tasks/neural_model_def")

model_neural_test(X_test, Y_test)

#model = baseline_model()

#from keras.utils import plot_model
#plot_model(model, to_file='/Users/Nurislam/PycharmProjects/diplom_project_2/model.png')

#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

#results = cross_val_score(estimator, X, Y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
