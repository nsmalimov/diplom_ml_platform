from sklearn import svm


def train(X, Y):
    clf = svm.SVC()
    clf.fit(X, Y)

    return clf


def test(model, X, Y):
    metrics = {}
    metrics['mean accuracy'] = model.score(X, Y)

    return metrics


def classify(model, features_arr):
    res_arr_class = [model.predict(i) for i in features_arr]
    res_arr_proba = [model.predict(i) for i in features_arr]
    return res_arr_class, res_arr_proba
