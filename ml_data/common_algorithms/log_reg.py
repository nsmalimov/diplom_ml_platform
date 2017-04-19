from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train(X, Y):
    lr = LogisticRegression()
    lr.fit(X, Y)

    return lr


def test(model, X, Y):
    metrics = {}
    metrics['mean accuracy'] = model.score(X, Y)

    return metrics


def classify(model, features_arr):
    res_arr = [model.predict(i) for i in features_arr]
    return res_arr
