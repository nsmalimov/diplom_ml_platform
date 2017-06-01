from sklearn.cluster.affinity_propagation_ import AffinityPropagation


def train(X):
    ap = AffinityPropagation()
    ap_fitted = ap.fit(X)

    return ap_fitted


# оценка через labels
def test(model, X, Y):
    metrics = None
    plots = None

    return metrics, plots


def get_labels(model, features_arr):
    labels = model.predict(features_arr)
    return labels
