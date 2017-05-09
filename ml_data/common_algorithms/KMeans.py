from sklearn.cluster import KMeans

def train(X):
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_fitted = kmeans.fit(X)

    return kmeans_fitted

# оценка через labels
def test(model, X, Y):
    metrics = None
    plots = None

    return metrics, plots


def get_labels(model, features_arr):
    return None, None
