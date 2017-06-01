import time

import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster.affinity_propagation_ import AffinityPropagation
from sklearn.cluster.birch import Birch
from sklearn.cluster.mean_shift_ import MeanShift

path = "/Users/Nurislam/Documents/ml_analysis_ws/test/data/"
#path = "/home/nur/PycharmProjects/ml_analysis_ws/test/data/"

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

    return predicted

    #print (Y_test)

    #print (Y_test)

    #plt = plot_3d_graphics(X_test, predicted)
    #plt.show()

def affinity_propagation(X, Y):
    ap = AffinityPropagation().fit(X)
    predicted = ap.predict(X)

    print (predicted)

    print (adjusted_rand_score(Y, predicted))

    return predicted

def birch(X, Y):
    brc = Birch().fit(X)
    predicted = brc.predict(X)

    print(predicted)

    print (adjusted_rand_score(Y, predicted))

    return predicted

def mean_shift(X, Y):
    ms = MeanShift(n_jobs=4).fit(X)
    predicted = ms.predict(X)

    print(predicted)

    print(adjusted_rand_score(Y, predicted))

def mini_batch_kmeans(X, Y):
    mbk = MiniBatchKMeans().fit(X)
    predicted = mbk.fit_predict(X)

    print (predicted)

    print(adjusted_rand_score(Y, predicted))

    return predicted

def get_count(pred_y, real_y):

    num_correct = 0

    for i in range(len(pred_y)):
        if (pred_y[i] == real_y[i]):
            num_correct += 1

    print (num_correct)
    print (len(real_y))
    print (num_correct / (len(real_y) + 0.0))

def get_count_in_one_cluster(pred_y, real_y):
    dict_in_one_pred = {}
    dict_in_one_real = {}

    num_correct = 0

    for i in range(len(pred_y)):
        if (pred_y[i] == real_y[i]):
            num_correct += 1

    print (num_correct)
    print (len(real_y))
    print (num_correct / (len(real_y) + 0.0))


import multiprocessing

num_cpu = multiprocessing.cpu_count()

print (num_cpu)

start_time = time.time()
print ("kmeans")
predicted = kmeans(X, Y)
get_count(predicted, Y)
print("--- %s seconds ---" % (time.time() - start_time))
print ("")

#exit()

#import numpy as np

#X_test = [[0,1,3], [2,4,5], [6,7,8]]
#Y_test = [1,2,3]
#X_test = np.array(X_test)
#print (X_test[:, 1], X_test[:, 0], X_test[:, 2])

#plt = plot_3d_graphics(X_test, Y_test)

#plt.show()

start_time = time.time()
print ("birch")
predicted = birch(X, Y)
get_count(predicted, Y)
print("--- %s seconds ---" % (time.time() - start_time))
print ("")

start_time = time.time()
print ("affinity_propagation")
predicted = affinity_propagation(X, Y)
get_count(predicted, Y)
print("--- %s seconds ---" % (time.time() - start_time))
print ("")

# long
# start_time = time.time()
# print ("mean_shift")
# predicted = mean_shift(X, Y)
# get_count(predicted, Y)
# print("--- %s seconds ---" % (time.time() - start_time))
# print ("")

print ("mini_batch_kmeans")
start_time = time.time()
predicted = mini_batch_kmeans(X, Y)
get_count(predicted, Y)
print("--- %s seconds ---" % (time.time() - start_time))

#\item k-means \cite{md};
#\item Affinity Propagation \cite{md};
#\item Birch \cite{md}.
#\end{enumerate}