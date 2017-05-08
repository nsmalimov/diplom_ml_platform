
def prepare_data():
    data = []
    f = open("/Users/Nurislam/PycharmProjects/diplom_ml_platform/test/data/digits.csv", "r")

    for i in f.readlines():
        s = i.replace("\n", "")
        s_split = s.split(",")
        s_new = ",".join(s_split[1:]) + "," + s_split[0]
        data.append(s_new)
    f.close()

    f = open("/Users/Nurislam/PycharmProjects/diplom_ml_platform/test/data/digits.csv", "w")

    for i in data:
        f.write(i + "\n")
    f.close()

prepare_data()