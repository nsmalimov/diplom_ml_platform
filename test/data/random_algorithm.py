import random

class RandomClassifier():
    def __init__(self):
        pass

    def fit(self, X, Y):
        pass

    def predict(self, X):
        return random.randint(0,1)

    def predict_proba(self):
        pass


def train(X, Y):
    lr = RandomClassifier()
    lr.fit(X, Y)

    return lr


def test(model, X, Y):
    metrics = {}

    accuracy = 0

    predicted_arr = []

    for i in X:
        predicted_arr.append(model.predict(i))

    for i in range(len(predicted_arr)):
        if predicted_arr[i] == Y[i]:
            accuracy += 1

    accuracy = sum(accuracy) / (len(predicted_arr) + 0.0)
    metrics['mean accuracy'] = accuracy
    return metrics


def classify(model, features_arr):
    res_arr_class = [model.predict(i) for i in features_arr]
    res_arr_proba = [model.predict(i) for i in features_arr]
    return res_arr_class, res_arr_proba
