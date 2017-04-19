# TODO доделать

from sklearn.metrics import hamming_loss
from sklearn.metrics import log_loss

predetermined_metrics_dict = {"log_loss": [lambda x, y: log_loss(x, y), "proba"],
                              "hamming_loss": [lambda x, y: hamming_loss(x, y), "class"]}


def get_predetermined_metrics(y_real_label, y_class_predict, y_proba_predict=None):
    metrics = {}

    for i in predetermined_metrics_dict:
        metrics[i] = predetermined_metrics_dict[i][0](x, y)

    return metrics
