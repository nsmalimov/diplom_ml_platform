import os


def get_main_project_dir():
    result = ""

    dir = os.path.dirname(os.path.abspath(__file__))

    dir_split = dir.split("/")

    for i in dir_split:
        result += i + "/"
        if (i == project_name):
            break

    return result


project_name = "diplom_ml_platform"
main_project_dir = get_main_project_dir()

ml_path = get_main_project_dir() + "ml_data/"
ml_path = ml_path.replace("/app/util", "")


def create_project_files(project_name):
    project_name = "project_" + str(project_name)
    model_filename = ml_path + project_name
    create_folder(model_filename)

    data_filename = ml_path + project_name + "/data"
    create_folder(data_filename)

    algorithm_filename = ml_path + project_name + "/algorithms"
    create_folder(algorithm_filename)

    models_filename = ml_path + project_name + "/models"
    create_folder(models_filename)

    models_filename = ml_path + project_name + "/results"
    create_folder(models_filename)
    models_filename = ml_path + project_name + "/results/images"
    create_folder(models_filename)


def load_object(project_name, object_type, filename):
    project_name = "project_" + str(project_name)
    filepath = ml_path + project_name + "/" + object_type + "/" + filename
    return filepath


def delete_project_files(project_name):
    project_name = "project_" + str(project_name)
    os.removedirs(ml_path + project_name)
    os.removedirs(ml_path + project_name + "/data")
    os.removedirs(ml_path + project_name + "/algorithms")
    os.removedirs(ml_path + project_name + "/models")
    os.removedirs(ml_path + project_name + "/results")


def save_file(file, filename, project_name, object_type):
    project_name = "project_" + str(project_name)
    try:
        file.save(ml_path + project_name + "/" + object_type + "/" + filename)
    except:
        create_project_files(project_name)


def delete_file(filename, project_name, type):
    project_name = "project_" + str(project_name)
    try:
        os.remove(ml_path + project_name + "/" + type + "/" + filename)
    except:
        pass


def create_folder(folder_name):
    if not (os.path.isdir(folder_name)):
        os.makedirs(folder_name)
