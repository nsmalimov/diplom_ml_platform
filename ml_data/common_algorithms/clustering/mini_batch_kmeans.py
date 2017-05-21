from sklearn.cluster.k_means_ import MiniBatchKMeans


def train(X):
    ap = MiniBatchKMeans().fit(X)

    return ap


# оценка через labels
def test(model, X, Y):
    metrics = None
    plots = None

    return metrics, plots


def get_labels(model, features_arr):
    labels = model.predict(features_arr)
    return labels
