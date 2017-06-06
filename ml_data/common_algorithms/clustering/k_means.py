from sklearn.cluster import KMeans
from app.init_server import num_cpu
from app.util.plot import plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd


def plot_hist(clusters):
    plt.clf()
    plt.cla()
    plt.close()

    data_frame = pd.DataFrame(clusters)
    plt.figure()
    data_frame.hist()

    return plt

def train(X):
    # TODO
    # а сколько у нас кластеров?
    kmeans = KMeans(n_jobs=num_cpu)
    kmeans_fitted = kmeans.fit(X)

    return kmeans_fitted


# оценка через labels
def test(model, X, Y):
    metrics = None
    plots = {}
    plots["clusters_hist"] = plot_hist(model.predict(X))

    return metrics, plots


def get_labels(model, features_arr):
    labels = model.predict(features_arr)
    return labels
