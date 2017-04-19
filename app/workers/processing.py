import hashlib
import importlib.util
import json
import os
import pickle

from sklearn.model_selection import train_test_split

from app.util.funcs import ml_path


def read_data(data, project):
    X, Y = [], []
    f = open(ml_path + str(project.id) + "/data/" + data.filename)

    # TODO текстовый таргет

    # TODO разделитель, целевая переменная, тип данных
    for i in f.readlines():
        s = i.replace("\n", "")
        s_split = s.split(",")

        arr = s_split[:-1]

        arr = [float(j) for j in arr]

        X.append(arr)
        Y.append(float(s_split[-1]))

    f.close()

    return X, Y


def import_alg(path_to_alg):
    # python 3.5 ?
    spec = importlib.util.spec_from_file_location("module.name", path_to_alg)
    alg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alg)

    return alg


def train_model(algorithm, X, Y, project):
    if algorithm.preloaded:
        path_to_alg = algorithm.filename
    else:
        path_to_alg = ml_path + str(project.id) + "/algorithms/" + algorithm.filename

    alg = import_alg(path_to_alg)

    model = alg.train(X, Y)

    return model


def get_hash_by_data_alg(data, algorithm, project):
    if algorithm.preloaded:
        path_to_alg = algorithm.filename
    else:
        path_to_alg = ml_path + str(project.id) + "/algorithms/" + algorithm.filename

    f = open(path_to_alg, "r")
    algorithm_code = f.read()
    f.close()

    f = open(ml_path + str(project.id) + "/data/" + data.filename)
    data_from_file = f.read()
    f.close()

    all = algorithm_code + data_from_file

    hash_md5 = hashlib.md5(all.encode('utf-8')).hexdigest()[:32]

    return hash_md5


def check_model_exist(hash_md5, project):
    path_model = ml_path + str(project.id) + "/models/" + hash_md5
    return os.path.isfile(path_model)


def save_model(model, project, hash_md5):
    path_model = ml_path + str(project.id) + "/models/" + hash_md5
    f = open(path_model, "wb")
    pickle.dump(model, f)
    f.close()


def train_and_save_model(data, algorithm, project):
    hash_md5 = get_hash_by_data_alg(data, algorithm, project)

    if not (check_model_exist(hash_md5, project)):
        X, Y = read_data(data, project)
        model = train_model(algorithm, X, Y, project)
        save_model(model, project, hash_md5)
        print("train and save")
    else:
        print("model already exist")


def read_model(project, hash_md5):
    path_model = ml_path + str(project.id) + "/models/" + hash_md5

    f = open(path_model, "rb")
    model = pickle.load(f)
    f.close()

    return model


def predict():
    pass


# возможно стоит тоже по хеш сумме проверять уже существующий
def get_metrics_plots_from_alg(data, algorithm, project):
    X, Y = read_data(data, project)

    hash_md5 = get_hash_by_data_alg(data, algorithm, project)
    model = read_model(project, hash_md5)

    _, X_test, _, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    if algorithm.preloaded:
        path_to_alg = algorithm.filename
    else:
        path_to_alg = ml_path + str(project.id) + "/algorithms/" + algorithm.filename

    print (path_to_alg)

    alg = import_alg(path_to_alg)

    metrics, plots = alg.test(model, X_test, y_test)

    path_to_plots = ml_path + str(project.id) + "/results/images/"

    plots_res = {}

    for i in plots:
        #plots[i].savefig(path_to_plots + i + ".png")
        plots_res[i] = "/Users/Nurislam/PycharmProjects/diplom_ml_platform/ml_data/1/results/images/roc_auc.png "#path_to_plots + i + ".png"

        plots_res[i] = plots_res[i].replace("/", ":")

        plots_res[i] = "/imageplot" + "/" + plots_res[i]

        print (plots_res[i])

    return metrics, plots_res


def start_processing_func(project, result_type, data, algorithm, analys_classif):
    type = result_type.name

    metrics1 = {}
    metrics2 = {}

    plots1 = {}
    plots2 = {}

    if type == "train_save_metrics_graphics":
        train_and_save_model(data, algorithm, project)

        # метрики или свои на выбор или заранее заданные
        metrics1, plots1 = get_metrics_plots_from_alg(data, algorithm, project)
        #metrics2 = get_metrics_predetermined(data, algorithm, project)

    data = {}
    data['type'] = 'train_save_metrics_graphics'

    data['metrics'] = metrics1
    data['metrics'].update(metrics2)

    data['img'] = plots1
    data['img'].update(plots2)

    res_json = json.dumps(data)

    return res_json
