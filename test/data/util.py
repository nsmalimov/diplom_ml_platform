path = "/Users/Nurislam/PycharmProjects/diplom_ml_platform/test/data/"

X, Y = [], []

f = open(path + "digits_old.csv", "r")

for i in f.readlines():
    s = i.replace("\n", "")
    s_split = s.split(",")

    X.append([float(i) for i in s_split[:-1]])
    Y.append(s_split[-1])

f.close()

delimiter = int(len(X)/100*10)

X = X[:delimiter]

Y = Y[:delimiter]

f = open(path + "digits.csv", "w")

for i in range(len(X)):
    arr = [str(j) for j in X[i]]
    s = ",".join(arr) + ","
    s += Y[i]

    f.write(s + "\n")

f.close()