from sklearn.cluster.affinity_propagation_ import AffinityPropagation

def train(X):
    ap = AffinityPropagation().fit(X)

    return ap

# оценка через labels
def test(model, X, Y):
    metrics = None
    plots = None

    return metrics, plots


def get_labels(model, features_arr):
    return None
