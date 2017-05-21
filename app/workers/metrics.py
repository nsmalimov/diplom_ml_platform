# TODO доделать
from sklearn.metrics import log_loss
from sklearn.metrics.cluster.supervised import adjusted_rand_score


def mean_score(y_true, y_pred):
    correct = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1

    mean_score = correct / (len(y_pred) + 0.0)

    return mean_score


predetermined_metrics_dict_classif = {"log_loss": [lambda y_true, y_pred: log_loss(y_true, y_pred), "proba"],
                              "mean_score": [lambda y_true, y_pred: mean_score(y_true, y_pred), "class"]}

predetermined_metrics_dict_cluster = {"adjusted_rand_score":
                                          lambda y_true, y_pred: adjusted_rand_score(y_true, y_pred)}

# эти метрики всегда могут вычисляться
def get_predetermined_metrics_classif(y_real_label, y_class_predict, y_proba_predict=None):
    metrics = {}

    for i in predetermined_metrics_dict_classif:
        need_type = predetermined_metrics_dict_classif[i][1]
        func = predetermined_metrics_dict_classif[i][0]
        if need_type == "proba" and not(y_proba_predict is None):
            metrics[i] = func(y_real_label, y_class_predict)
        if need_type == "class":
            metrics[i] = func(y_real_label, y_class_predict)

    return metrics

def get_predetermined_metrics_cluster(real_clusters_arr, clusters_from_alg):
    metrics = {}

    for i in predetermined_metrics_dict_cluster:
        func = predetermined_metrics_dict_cluster[i]
        metrics[i] = func(real_clusters_arr, clusters_from_alg)

    return metrics