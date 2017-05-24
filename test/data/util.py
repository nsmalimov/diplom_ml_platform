# path = "/Users/Nurislam/Documents/ml_analysis_ws/diplom_ml_platform/test/data/"
#
# X, Y = [], []
#
# f = open(path + "digits_old.csv", "r")
#
# for i in f.readlines():
#     s = i.replace("\n", "")
#     s_split = s.split(",")
#
#     X.append([float(i) for i in s_split[:-1]])
#     Y.append(s_split[-1])
#
# f.close()
#
# delimiter = int(len(X)/100*10)
#
# X = X[:delimiter]
#
# Y = Y[:delimiter]
#
# f = open(path + "digits.csv", "w")
#
# for i in range(len(X)):
#     arr = [str(j) for j in X[i]]
#     s = ",".join(arr) + ","
#     s += Y[i]
#
#     f.write(s + "\n")
#
# f.close()

import pandas as pd
from app.util.plot import plt

labels = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,6,6,7,5,4]

data_frame = pd.DataFrame(labels)

print (data_frame)

plt.figure()
data_frame.hist()
plt.show()

