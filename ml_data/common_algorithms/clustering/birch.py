from sklearn.cluster.birch import Birch
from sklearn.metrics.cluster.supervised import completeness_score


def train(X):
    # TODO
    # а сколько у нас кластеров?

    brc = Birch().fit(X)

    return brc


# оценка через labels
def test(model, X, Y):
    metrics = {}
    labels = model.predict(X)
    metrics["completeness_score"] = completeness_score(Y, labels)
    plots = None

    return metrics, plots


def get_labels(model, features_arr):
    labels = model.predict(features_arr)
    return labels
