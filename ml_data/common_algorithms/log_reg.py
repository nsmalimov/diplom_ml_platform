from app.util.plot import plt
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def train(X, Y):
    lr = LogisticRegression(penalty='l1')
    lr.fit(X, Y)

    return lr


def test(model, X, Y):
    metrics = {}
    metrics['mean accuracy'] = model.score(X, Y)

    X = X + X + X
    Y = Y + Y + Y

    plots = {}

    title = "Learning Curves"
    cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    plots['learning curves'] = plot_learning_curve(model, title, X, Y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    return metrics, plots


def classify(model, features_arr):
    res_arr_class = [model.predict(i) for i in features_arr]
    res_arr_proba = [model.predict(i) for i in features_arr]
    return res_arr_class, res_arr_proba
