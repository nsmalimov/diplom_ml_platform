import random


class RandomClassifier():
    def __init__(self, num_classes=2, start_num=0):
        self.num_classes = num_classes - 1
        self.start_num = start_num

    def fit(self, X, Y):
        pass

    def predict(self, X):
        lst = [i for i in range(int(self.start_num), int(self.num_classes) + 1)]
        return random.choice(lst)

    def predict_proba(self):
        pass


def train(X, Y):
    Y_new = list(set(Y))
    num_classes = len(Y_new)
    start_num = min(Y_new)

    lr = RandomClassifier(num_classes, start_num)
    lr.fit(X, Y)

    return lr


def test(model, X, Y):
    metrics = {}
    plots = {}

    return metrics, plots


def classify(model, features_arr):
    res_arr_class = [model.predict(i) for i in features_arr]
    return res_arr_class, None
