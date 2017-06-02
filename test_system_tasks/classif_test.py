from sklearn.cross_validation import train_test_split
from sklearn.metrics.classification import log_loss

path = "/Users/Nurislam/Documents/ml_analysis_ws/files_for_test/"
#path = "/Users/Nurislam/Documents/ml_analysis_ws/test/data/"

#filename = "commonData.csv"
filename = "wine.csv"

def prepare_data():
    data = []
    f = open(path + filename, "r")

    for i in f.readlines():
        s = i.replace("\n", "")
        s_split = s.split(",")
        s_new = ",".join(s_split[1:]) + "," + s_split[0]
        data.append(s_new)
    f.close()

    f = open(path + filename, "w")

    for i in data:
        f.write(i + "\n")
    f.close()

#prepare_data()

X, Y = [], []

f = open(path + filename, "r")

for i in f.readlines():
    s = i.replace("\n", "")
    s_split = s.split(",")

    X.append([float(i) for i in s_split[:-1]])
    Y.append(int(s_split[-1]))

f.close()

def mean_score(y_true, y_pred):
    correct = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1

    mean_score = correct / (len(y_pred) + 0.0)

    return mean_score


import ml_data.common_algorithms.classification.random_forest

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

model = ml_data.common_algorithms.classification.random_forest.train(X_train, Y_train)
print (model.score(X_test, Y_test))

print (log_loss(model.predict_proba(X_test), Y_test))

# import ml_data.common_algorithms.classification.log_reg
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# model1 = ml_data.common_algorithms.classification.log_reg.train(X_train, Y_train)
# print (model1.score(X_test, Y_test))
#
# import ml_data.common_algorithms.classification.svm
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# model2 = ml_data.common_algorithms.classification.svm.train(X_train, Y_train)
# print (model2.score(X_test, Y_test))
#
# import files_for_test.random_algorithm
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# model3 = files_for_test.random_algorithm.train(X_train, Y_train)
# print (files_for_test.random_algorithm.test(model3, X_test, Y_test))

# predicted = files_for_test.random_algorithm.classify(model3, X_test)[0]
#
# print (predicted)
#
# print (mean_score(predicted, Y_test))

# neural
# import files_for_test.keras_classif
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# model4 = files_for_test.keras_classif.train(X_train, Y_train)
# predicted_class, predicted_proba = files_for_test.keras_classif.classify(model4, X_test)
#
# print (predicted_proba)
# print (mean_score(predicted_class, Y_test))

