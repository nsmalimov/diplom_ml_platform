# TODO доделать
from sklearn.metrics import log_loss

def mean_score(y_true, y_pred):
    correct = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1

    mean_score = correct / (len(y_pred) + 0.0)

    return mean_score


predetermined_metrics_dict = {"log_loss": [lambda y_true, y_pred: log_loss(y_true, y_pred), "proba"],
                              "mean_score": [lambda y_true, y_pred: mean_score(y_true, y_pred), "class"]}


def get_predetermined_metrics(y_real_label, y_class_predict, y_proba_predict=None):
    metrics = {}

    for i in predetermined_metrics_dict:
        need_type = predetermined_metrics_dict[i][1]
        func = predetermined_metrics_dict[i][0]
        if need_type == "proba" and not(y_proba_predict is None):
            metrics[i] = func(y_real_label, y_class_predict)
        if need_type == "class":
            metrics[i] = func(y_real_label, y_class_predict)

    return metrics
