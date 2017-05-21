from sklearn.cluster import KMeans
from app.init_server import num_cpu

def train(X):
    # TODO
    # а сколько у нас кластеров?
    kmeans = KMeans(n_jobs = num_cpu)
    kmeans_fitted = kmeans.fit(X)

    return kmeans_fitted

# оценка через labels
def test(model, X, Y):
    metrics = None
    plots = None

    return metrics, plots


def get_labels(model, features_arr):
    labels = model.predict(features_arr)
    return labels
