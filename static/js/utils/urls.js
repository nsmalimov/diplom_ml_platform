var baseUrl = location.protocol+'//'+location.hostname+(location.port ? ':'+location.port: '') + "/";

var urlsList = {
    data: {
        upload_one: "data_upload",
        load_all: "data_load_all",
        load_all_by_project: "data_load_all_by_project",
        delete: "delete_object",
        load_all_by_project_and_task_type: "data_load_all_by_project_and_task_type"
    },

    project: {
        create: "project_create",
        load_all: "project_load_all",
        delete: "delete_object"
    },

    algorithm: {
        upload_one: "algorithm_upload",
        load_all: "algorithm_load_all",
        load_manual_by_project: "algorithm_load_all_by_project_by_type",
        load_all_by_project: "algorithm_load_all_by_project",
        delete: "delete_object",
        load_all_common: "algorithm_load_common"
    },

    analysClassif: {
        load_all_result_types: "analys_classif_all_result_types",
        start_processing: "analys_classif_start_processing"
    }
};