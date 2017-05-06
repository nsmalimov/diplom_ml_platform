import pickle
#
# f = open("/Users/Nurislam/PycharmProjects/diplom_ml_platform/111", "rb")
# test_X, test_Y = pickle.load(f)
#
# f.close()
#
# f = open("/Users/Nurislam/PycharmProjects/diplom_ml_platform/ml_data/1/models/162610950", "rb")
#
# model = pickle.load(f)
#
# f.close()
#
# print (model)
#
# from ml_data.common_algorithms.log_reg import test
# #import ml_data.common_algorithms.log_reg.train
#
# metrics, plots = test(model, test_X, test_Y)
#
# print (metrics)
#
# f = open("/Users/Nurislam/PycharmProjects/diplom_ml_platform/ml_data/1/models/32032800", "rb")
#
# model = pickle.load(f)
#
# f.close()
#
# print (type(model))
#
# from ml_data.common_algorithms.svm import test
# #import ml_data.common_algorithms.log_reg.train
#
# metrics, plots = test(model, test_X, test_Y)
#
# print (metrics)
#
#

from test1.rm import RandomClassifier

f = open("/Users/Nurislam/PycharmProjects/diplom_ml_platform/ml_data/1/models/kkk", "wb")
r = RandomClassifier()

print (r)

pickle.dump(r, f)

f.close()
