import numpy as np
from sklearn import svm


def train(X, Y):
    clf = svm.SVC()
    clf.fit(X, Y)

    return clf


def test(model, X, Y):
    metrics = {}
    metrics = None
    # metrics['mean accuracy'] = model.score(X, Y)

    plots = None

    return metrics, plots


def classify(model, features_arr):
    res_arr_class = [model.predict(np.array(i).reshape(1, -1)) for i in features_arr]
    res_arr_proba = None
    return res_arr_class, res_arr_proba
