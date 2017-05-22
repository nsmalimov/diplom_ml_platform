from sklearn import datasets

from sklearn.cluster.affinity_propagation_ import AffinityPropagation
from sklearn.cluster.birch import Birch
from sklearn.cross_validation import train_test_split
from sklearn.cluster import MeanShift, MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import time
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

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


def plot_3d_graphics(X, clusters):
    X = np.array(X)
    from app.util.plot import plt
    fig = plt.figure(0, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig)

    plt.cla()

    print (X[:, 12], X[:, 5], X[:, 8])

    ax.scatter(X[:, -1], X[:, -2], X[:, -3], c=clusters)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('X[0]')
    ax.set_ylabel('X[1]')
    ax.set_zlabel('X[2]')

    return plt

from sklearn.metrics import adjusted_rand_score

#iris = datasets.load_iris()
#X = iris.data
#Y = iris.target

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

Y = [int(i) for i in Y]

def kmeans(X, Y):
    kmeans = KMeans(n_jobs=4).fit(X)
    predicted = kmeans.predict(X)

    print (predicted)

    print (adjusted_rand_score(Y, predicted))

    #print (Y_test)

    #print (Y_test)

    #plt = plot_3d_graphics(X_test, predicted)
    #plt.show()

# def affinity_propagation(X_train, X_test, Y_train, Y_test):
#     ap = AffinityPropagation().fit(X_train)
#     predicted = ap.predict(X_test)
#
#     print (predicted)
#
#     print (adjusted_rand_score(Y_test, predicted))

def birch(X, Y):
    brc = Birch().fit(X)
    predicted = brc.predict(X)

    print(predicted)

    print (adjusted_rand_score(Y, predicted))

# def mean_shift(X_train, X_test, Y_train, Y_test):
#     ms = MeanShift(n_jobs=4).fit(X_train)
#     predicted = ms.predict(X_test)
#
#     print(predicted)
#
#     print(adjusted_rand_score(Y_test, predicted))

def mini_batch_kmeans(X, Y):
    mbk = MiniBatchKMeans().fit(X)
    predicted = mbk.fit_predict(X)

    print (predicted)

    print(adjusted_rand_score(Y, predicted))

import multiprocessing

num_cpu = multiprocessing.cpu_count()

print (num_cpu)

start_time = time.time()
kmeans(X, Y)
print("--- %s seconds ---" % (time.time() - start_time))

#import numpy as np

#X_test = [[0,1,3], [2,4,5], [6,7,8]]
#Y_test = [1,2,3]
#X_test = np.array(X_test)
#print (X_test[:, 1], X_test[:, 0], X_test[:, 2])

#plt = plot_3d_graphics(X_test, Y_test)

#plt.show()

start_time = time.time()
birch(X, Y)
print("--- %s seconds ---" % (time.time() - start_time))
#
# start_time = time.time()
# affinity_propagation(X_train, X_test, Y_train, Y_test)
# print("--- %s seconds ---" % (time.time() - start_time))
#
# start_time = time.time()
# mean_shift(X_train, X_test, Y_train, Y_test)
# print("--- %s seconds ---" % (time.time() - start_time))
#
start_time = time.time()
mini_batch_kmeans(X, Y)
print("--- %s seconds ---" % (time.time() - start_time))

#\item k-means \cite{md};
#\item Affinity Propagation \cite{md};
#\item Birch \cite{md}.
#\end{enumerate}