import hashlib
import importlib.util
import json
import os
import pickle
import os
#from app.util.plot import plt
from imp import reload

from sklearn.model_selection import train_test_split

from app.util.funcs import ml_path
from app.workers.metrics import get_predetermined_metrics


def read_data(data, project):
    X, Y = [], []
    f = open(ml_path + "project_" + str(project.id) + "/data/" + data.filename)

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
    #path = "ml_data.project_1.algorithms.random_algorithm"
    #spec = importlib.util.spec_from_file_location(path, path_to_alg)
    #alg = importlib.util.module_from_spec(spec)
    #spec.loader.exec_module(alg)

    path = os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]
    path = "/".join(path)
    path = path_to_alg.replace(path, "")
    path = path.replace(".py", "")
    path = path.replace("/", ".")
    path = path[1:]

    alg = importlib.import_module(path)

    return alg


def train_model(algorithm, X, Y, project):
    if algorithm.preloaded:
        path_to_alg = algorithm.filename
    else:
        path_to_alg = ml_path + "project_" + str(project.id) + "/algorithms/" + algorithm.filename

    alg = import_alg(path_to_alg)

    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.33, random_state=42)

    model = alg.train(X_train, Y_train)

    return model


def get_hash_by_data_alg(data, algorithm, project):
    if algorithm.preloaded:
        path_to_alg = algorithm.filename
    else:
        path_to_alg = ml_path + "project_" + str(project.id) + "/algorithms/" + algorithm.filename

    f = open(path_to_alg, "r")
    algorithm_code = f.read()
    f.close()

    f = open(ml_path + "project_" + str(project.id) + "/data/" + data.filename)
    data_from_file = f.read()
    f.close()

    #all = algorithm_code + data_from_file

    mult_char = len(algorithm_code) * len(data_from_file)

    return str(mult_char)


def check_model_exist(hash, project):
    path_model = ml_path + "project_" + str(project.id) + "/models/" + hash
    return os.path.isfile(path_model)


def save_model(model, project, hash):
    path_model = ml_path + "project_" + str(project.id) + "/models/" + hash
    f = open(path_model, "wb")

    print (model)

    #import ml_data.project_1.algorithms.random_algorithm
    #reload(ml_data.project_1.algorithms.random_algorithm)

    pickle.dump(model, f)
    f.close()


def train_and_save_model(data, algorithm, project):
    hash = get_hash_by_data_alg(data, algorithm, project)

    print (hash)
    if not (check_model_exist(hash, project)):
        X, Y = read_data(data, project)
        model = train_model(algorithm, X, Y, project)
        save_model(model, project, hash)
        print("train and save")
    else:
        print("model already exist")


def read_model(project, hash):
    path_model = ml_path + "project_" + str(project.id) + "/models/" + hash

    f = open(path_model, "rb")
    model = pickle.load(f)
    f.close()

    return model


def predict():
    pass


# возможно стоит тоже по хеш сумме проверять уже существующий
def get_metrics_plots_from_alg(data, algorithm, project):
    X, Y = read_data(data, project)

    hash = get_hash_by_data_alg(data, algorithm, project)
    model = read_model(project, hash)

    _, X_test, _, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    if algorithm.preloaded:
        path_to_alg = algorithm.filename
    else:
        path_to_alg = ml_path + "project_" + str(project.id) + "/algorithms/" + algorithm.filename

    alg = import_alg(path_to_alg)

    metrics, plots = alg.test(model, X_test, y_test)

    path_to_plots = ml_path + "project_" + str(project.id) + "/results/images/"

    plots_res = {}

    y_class_predict, y_proba_predict = alg.classify(model, X_test)

    # TODO как закрыть plt
    # проверить как работает на сервере

    # какой-то график точно будет
    if plots is None:
        return metrics, plots_res, y_test, y_class_predict, y_proba_predict

    for i in plots:
        path = path_to_plots + i + ".png"
        path = path.replace(" ", "_")
        plots[i].savefig(path)
        #plt.close("all")

        plots_res[i] = path.replace("/", ":")

        plots_res[i] = "/imageplot/" + plots_res[i][1:]

    print (metrics, plots_res, y_test, y_class_predict, y_proba_predict)

    return metrics, plots_res, y_test, y_class_predict, y_proba_predict


def start_processing_func(project, result_type, data, algorithm, analys_classif):
    type = result_type.name

    metrics1 = {}

    plots1 = {}
    plots2 = {}

    if type == "train_save_metrics_graphics":
        train_and_save_model(data, algorithm, project)

        # метрики или свои на выбор или заранее заданные
        metrics1, plots1, y_real_label, y_class_predict, y_proba_predict = get_metrics_plots_from_alg(data, algorithm, project)
        metrics2 = get_predetermined_metrics(y_real_label, y_class_predict, y_proba_predict)

    data = {}
    data['type'] = 'train_save_metrics_graphics'

    data['metrics'] = metrics1
    data['metrics'].update(metrics2)

    data['img'] = plots1
    data['img'].update(plots2)

    res_json = json.dumps(data)

    return res_json
