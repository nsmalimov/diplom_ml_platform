import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.classification import hamming_loss

from app.init_server import num_cpu


def train(X, Y):
    rf = RandomForestClassifier(n_jobs=num_cpu)
    rf.fit(X, Y)

    return rf


def test(model, X, Y):
    metrics = {}
    # metrics['mean accuracy'] = model.score(X, Y)

    predicted_proba = model.predict(np.array(X))

    # precision recall f1 = 0 ...

    metrics['hamming loss'] = hamming_loss(Y, predicted_proba)

    plots = None

    return metrics, plots


# изменить
def classify(model, features_arr):
    res_arr_class = [model.predict(np.array(i).reshape(1, -1)) for i in features_arr]
    res_arr_proba = [model.predict_proba(np.array(i).reshape(1, -1)) for i in features_arr]
    return res_arr_class, res_arr_proba
