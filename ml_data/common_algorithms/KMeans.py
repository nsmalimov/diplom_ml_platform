from sklearn.cluster import KMeans

def train(X):
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_fitted = kmeans.fit(X)

    return kmeans_fitted


def test(model, X, Y):
    metrics = None
    plots = None

    return metrics, plots


def get_labels(model, features_arr):
    res_arr_class = [model.predict(np.array(i).reshape(1, -1)) for i in features_arr]
    res_arr_proba = [model.predict_proba(np.array(i).reshape(1, -1)) for i in features_arr]
    return res_arr_class, res_arr_proba
