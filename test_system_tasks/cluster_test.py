from sklearn import datasets

from sklearn.cluster.affinity_propagation_ import AffinityPropagation
from sklearn.cluster.birch import Birch
from sklearn.cross_validation import train_test_split
from sklearn.cluster import MeanShift, MiniBatchKMeans
from sklearn.cluster import DBSCAN
import time

path = "/Users/Nurislam/PycharmProjects/diplom_ml_platform/test/data/"
#path = "/home/nur/PycharmProjects/diplom_ml_platform/test/data/"

def prepare_data():
    data = []
    f = open(path + "digits.csv", "r")

    for i in f.readlines():
        s = i.replace("\n", "")
        s_split = s.split(",")
        s_new = ",".join(s_split[1:]) + "," + s_split[0]
        data.append(s_new)
    f.close()

    f = open(path + "digits.csv", "w")

    for i in data:
        f.write(i + "\n")
    f.close()

#prepare_data()

X, Y = [], []

f = open(path + "digits.csv", "r")

for i in f.readlines():
    s = i.replace("\n", "")
    s_split = s.split(",")

    X.append([float(i) for i in s_split[:-1]])
    Y.append(s_split[-1])

f.close()

#delimiter = int(len(X)/100*10)

#X = X[:delimiter]

#Y = Y[:delimiter]

print (len(X))
print (len(Y))

# dict_1 = {}
#
# for i in Y:
#     if i in dict_1:
#         dict_1[i] += 1
#     else:
#         dict_1[i] = 1
#
# for i in dict_1:
#     print (str(i) + " " + str(dict_1[i]))
#
# exit()

from sklearn.metrics import adjusted_rand_score

#iris = datasets.load_iris()
#X = iris.data
#Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

Y_test = [int(i) for i in Y_test]

def kmeans(X_train, X_test, Y_train, Y_test):
    from sklearn.cluster import KMeans
    import numpy as np
    kmeans = KMeans(n_clusters=10, random_state=0, n_jobs=4).fit(X_train)
    predicted = kmeans.predict(X_test)

    print (predicted)
    print (adjusted_rand_score(Y_test, predicted))
    print (Y_test)

def affinity_propagation(X_train, X_test, Y_train, Y_test):
    ap = AffinityPropagation().fit(X_train)
    predicted = ap.predict(X_test)

    print (predicted)

    print (adjusted_rand_score(Y_test, predicted))

def birch(X_train, X_test, Y_train, Y_test):
    brc = Birch(n_clusters=10).fit(X_train)
    predicted = brc.predict(X_test)

    print (adjusted_rand_score(Y_test, predicted))

def mean_shift(X_train, X_test, Y_train, Y_test):
    ms = MeanShift(n_jobs=4).fit(X_train)
    predicted = ms.predict(X_test)

    print(predicted)

    print(adjusted_rand_score(Y_test, predicted))

def mini_batch_kmeans(X_train, X_test, Y_train, Y_test):
    mbk = MiniBatchKMeans().fit(X_train)

    predicted = mbk.fit_predict(X_test)

    print (predicted)

    print(adjusted_rand_score(Y_test, predicted))

import multiprocessing

num_cpu = multiprocessing.cpu_count()

print (num_cpu)

# start_time = time.time()
# kmeans(X_train, X_test, Y_train, Y_test)
# print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# birch(X_train, X_test, Y_train, Y_test)
# print("--- %s seconds ---" % (time.time() - start_time))
#
# start_time = time.time()
# affinity_propagation(X_train, X_test, Y_train, Y_test)
# print("--- %s seconds ---" % (time.time() - start_time))
#
# start_time = time.time()
# mean_shift(X_train, X_test, Y_train, Y_test)
# print("--- %s seconds ---" % (time.time() - start_time))
#
# start_time = time.time()
# mini_batch_kmeans(X_train, X_test, Y_train, Y_test)
# print("--- %s seconds ---" % (time.time() - start_time))

#\item k-means \cite{md};
#\item Affinity Propagation \cite{md};
#\item Birch \cite{md}.
#\end{enumerate}