from app.util.plot import plt
import numpy as np
from app.util.funcs import ml_path


predetermined_plots_dict_classes = {"classes_by_first_two_features": lambda var_1, var_2:
                                          get_plot_classes_by_first_two_features(var_1, var_2)}

predetermined_plots_dict_cluster = {"clusters_by_first_two_features": lambda var_1, var_2:
                                          get_plot_clusters_by_first_two_features(var_1, var_2)}

def get_predetermined_plots_classif(X, y_class_predict, project_id):
    plots = {}

    for i in predetermined_plots_dict_classes:
        plots[i] = predetermined_plots_dict_classes[i](X, y_class_predict)

    path_to_plots = ml_path + "project_" + str(project_id) + "/results/images/"

    plots_res = {}

    print (plots)

    for i in plots:
        path = path_to_plots + i + ".png"
        path = path.replace(" ", "_")
        plots[i].savefig(path)

        plots_res[i] = path.replace("/", ":")

        plots_res[i] = "/imageplot/" + plots_res[i][1:]

    return plots_res

def get_predetermined_plots_cluster(X, y_clusters_predict, project_id):
    plots = {}

    for i in predetermined_plots_dict_cluster:
        plots[i] = predetermined_plots_dict_cluster[i](X, y_clusters_predict)

    path_to_plots = ml_path + "project_" + str(project_id) + "/results/images/"

    plots_res = {}

    for i in plots:
        path = path_to_plots + i + ".png"
        path = path.replace(" ", "_")
        plots[i].savefig(path)

        plots_res[i] = path.replace("/", ":")

        plots_res[i] = "/imageplot/" + plots_res[i][1:]

    return plots_res

def get_plot_classes_by_first_two_features(X, predicted_class):
    plt.title("Classes by first two features")
    plt.xlabel("X[0]")
    plt.ylabel("X[1]")

    X = np.array(X)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=predicted_class)

    return plt

def get_plot_clusters_by_first_two_features(X, predicted_clusters):
    plt.title("Classes by first two features")
    plt.xlabel("X[0]")
    plt.ylabel("X[1]")

    X = np.array(X)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=predicted_clusters)

    return plt