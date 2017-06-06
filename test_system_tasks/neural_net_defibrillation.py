import copy
import random

import numpy as np
import scipy
import sklearn.metrics as m

import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential, save_model, load_model
from scipy.sparse import csr_matrix

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

X, Y = [], []

# f = open("/Users/Nurislam/PycharmProjects/diplom_project_2/data/concated/commonData.csv", "r")

path1 = "/Users/Nurislam/Documents/ml_analysis_ws/files_for_test/"
# path = "/home/nur/PycharmProjects/diplom_ml_platform/test/data/"

path2 = "/Users/Nurislam/Documents/ml_analysis_ws/test_system_tasks/neural_model_def"
# path2 = "/home/nur/PycharmProjects/diplom_ml_platform/test_system_tasks/neural_model_def"

f = open(path1 + "commonData.csv", "r")

for i in f.readlines():
    s = i.replace("\n", "")
    arr = s.split(",")

    X.append([float(i) for i in arr[:-1]])
    Y.append(int(arr[-1]))

f.close()


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

def proportional_split_test_train(X, Y):
    X_train, X_test, Y_train, Y_test = [], [], [], []

    split_num = 0.33
    each_elem_count = {}
    need_each_count = {}

    for i in Y:
        if not (i in each_elem_count):
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


def neural_model(X, y, retrain=True):
    if not (retrain):
        return load_model(path2)

    X = csr_matrix(X)

    y_train_new = []

    # for i in y:
    #    y_train_new.append(np.array([i]))

    y = [int(i) for i in y]

    y_train_new = np.array(y)

    # y_train_new[y_train_new == 1] = 0
    # y_train_new[y_train_new == 2] = 1
    # y_train_new[y_train_new == 3] = 2

    # one_hot_labels = keras.utils.to_categorical(y_train_new, num_classes=1)

    # tf.reset_default_graph()
    sess = tf.InteractiveSession()

    model = Sequential()

    model.add(Dense(15, input_dim=X[0].shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    # model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("start fit")

    print(y_train_new)
    # exit()
    model.fit(X.todense(), y_train_new, epochs=30, batch_size=5)

    save_model(model, path2)

    return model


def oversampling(X, Y):
    dict_count_each_class = {0: 0, 1: 0}

    for i in Y:
        dict_count_each_class[i] += 1

    max_count = max(dict_count_each_class.values())

    label_need = 0 if (list(dict_count_each_class.
                            keys())[list(dict_count_each_class.values()).
                       index(max_count)]) == 1 else 1

    arr = []
    count = 0

    for i in range(len(X)):
        label = Y[i]
        if label == label_need:
            arr.append(X[i])
        else:
            count += 1

    need_1 = count - len(arr)

    for i in range(need_1):
        r = random.randint(0, len(arr) - 1)
        choose_X = copy.deepcopy(arr[r])
        X.append(choose_X)
        Y.append(label_need)

    return X, Y


def undersampling(X, Y):
    with_zero = []
    with_one = []
    count_1 = 0
    count_0 = 0

    for i in range(len(X)):
        label = Y[i]
        if label == 1:
            with_one.append(X[i])
            count_1 += 1
        else:
            with_zero.append(X[i])
            count_0 += 1

    X_new = []
    Y_new = []

    for i in with_zero:
        if count_1 >= 0:
            X_new.append(i)
            Y_new.append(0)

        count_1 -= 1

    for i in with_one:
        X_new.append(i)
        Y_new.append(1)

    X = X_new
    Y = Y_new

    return X, Y


def expand_dataset(X, Y):
    X_new, Y_new = [], []

    for i in X:
        arr = i + i + i + i + i + i
        X_new.append(arr)

    X = X_new

    return X, Y


def model_neural_test(X_test, Y_test):
    model = load_model(path2)

    predict = model.predict(X_test)

    res_arr = []

    for i in predict:
        # print (i)
        # max_index = i.tolist().index(max(i))
        res_arr.append(round(i[0]))

    f1 = m.f1_score(Y_test, res_arr)
    print("f1 " + str(f1))

    precision = m.precision_score(Y_test, res_arr)
    print("precision " + str(precision))

    recall = m.recall_score(Y_test, res_arr)
    print("recall " + str(recall))

    count_correct = 0

    for i in range(len(res_arr)):
        print(str(res_arr[i]) + " " + str(Y_test[i]))
        count_correct += 1 if res_arr[i] == Y_test[i] else 0

    mean_score = (count_correct / (len(res_arr) + 0.0))
    print("mean score " + str(mean_score))


def read_wave_data():
    X, Y = [], []

    f = open(path1 + "lifepak_data", "r")

    for i in f.readlines():
        s = i.replace("\n", "")
        s_split = s.split(",")
        s_split = [float(j) for j in s_split]

        X.append(s_split[:-1])
        Y.append(s_split[-1])

    f.close()

    return X, Y


# X, Y = oversampling(X, Y)
# X, Y = undersampling(X, Y)

# print (Y)
# exit()
# X, Y = expand_dataset(X, Y)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

def get_cosine_dist():
    num = 0
    for i in X:
        if Y[num] != 0:
            num += 1
            continue
        print("")
        for j in X:
            print(scipy.spatial.distance.cosine(i, j))

        num += 1

def add_elements_rand(X, Y):
    dict_range_by_features_0 = {}
    dict_range_by_features_1 = {}

    for i in range(len(X)):
        num = 0

        if Y[i] == 0:
            for j in X[i]:
                if not(num in dict_range_by_features_0):
                    dict_range_by_features_0[num] = [j, j]
                else:
                    if dict_range_by_features_0[num][0] > j:
                        dict_range_by_features_0[num][0] = j

                    if dict_range_by_features_0[num][1] < j:
                        dict_range_by_features_0[num][1] = j
                num += 1
        else:
            for j in X[i]:
                if not(num in dict_range_by_features_1):
                    dict_range_by_features_1[num] = [j, j]
                else:
                    if dict_range_by_features_1[num][0] > j:
                        dict_range_by_features_1[num][0] = j

                    if dict_range_by_features_1[num][1] < j:
                        dict_range_by_features_1[num][1] = j
                num += 1

    return X, Y

X, Y = read_wave_data()

X, Y = oversampling(X, Y)

X, Y = add_elements_rand(X, Y)

X_train, X_test, Y_train, Y_test = proportional_split_test_train(X, Y)

model = neural_model(X_train, Y_train, True)

model_neural_test(X_test, Y_test)

# model = baseline_model()

# from keras.utils import plot_model

# print ("start plot")
# plot_model(model, to_file='/Users/Nurislam/Documents/ml_analysis_ws/model.png')

# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
