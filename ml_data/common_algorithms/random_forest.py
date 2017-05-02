from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

def train(X, Y):
    rf = RandomForestClassifier()
    rf.fit(X, Y)

    return rf


def test(model, X, Y):
    metrics = {}
    metrics['mean accuracy'] = model.score(X, Y)

    predicted_proba = model.predict_proba(X)

    # precision recall f1 = 0 ...
    metrics['log loss'] = log_loss(Y, predicted_proba)

    plots = None

    return metrics, plots


# изменить
def classify(model, features_arr):
    res_arr_class = [model.predict(i) for i in features_arr]
    res_arr_proba = [model.predict(i) for i in features_arr]
    return res_arr_class, res_arr_proba
