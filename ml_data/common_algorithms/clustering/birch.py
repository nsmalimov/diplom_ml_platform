from sklearn.cluster.birch import Birch

def train(X):
    # TODO
    # а сколько у нас кластеров?

    brc = Birch().fit(X)

    return brc

# оценка через labels
def test(model, X, Y):
    metrics = None
    plots = None

    return metrics, plots


def get_labels(model, features_arr):
    return None
