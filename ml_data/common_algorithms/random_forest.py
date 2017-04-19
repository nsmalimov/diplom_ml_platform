from sklearn.ensemble import RandomForestClassifier


def train(X, Y):
    rf = RandomForestClassifier()
    rf.fit(X, Y)

    return rf


def test(model, X, Y):
    metrics = {}
    metrics['mean accuracy'] = model.score(X, Y)

    return metrics



def classify(model, features_arr):
    res_arr = [model.predict(i) for i in features_arr]
    return res_arr
