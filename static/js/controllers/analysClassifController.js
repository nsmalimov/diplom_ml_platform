myApp.controller("analysClassifController", function ($scope, $http) {

    $scope.selectedObjects = {
        "selectedProject": null,
        "selectedRecord": null,
        "selectedAlgorithm": null,
        "selectedResultType": null
    };

    $scope.selectType = null;

    $scope.selectedProject = null;
    $scope.selectedRecord = null;
    $scope.selectedAlgorithm = null;
    $scope.selectedResultType = null;

    $scope.findBest = false;

    $scope.bestClassif = null;
    $scope.bestCluster = null;

    $scope.type = null;
    $scope.metrics = null;
    $scope.resultImages = null;

    // TODO выбор нескольких алгоритмов
    $scope.selectedAlgorithmsArray = [];

    $scope.commonAlgorithms = null;

    $scope.algorithmsByProjectAndCommon = null;

    $scope.resultTypes = null;

    $scope.res = null;

    $scope.loadAllProjects = function () {
        $http({
            method: 'GET',
            url: urlsList.project.load_all
        }).then(function successCallback(response) {
            $scope.allProjects = response.data;
        }, function errorCallback(response) {
        });
    };

    $scope.loadAllProjects();

    $scope.loadAllDataByProjectId = function (project_id) {
        if (project_id) {
            $http({
                method: 'POST',
                dataType: 'json',
                url: urlsList.data.load_all_by_project,
                data: JSON.stringify({project_id: project_id}),
                contentType: 'application/json'
            }).then(function successCallback(response) {
                $scope.allRecordsByProjectId = response.data;
            }, function errorCallback(response) {
            })
        }
    };

    $scope.loadAllAlgorithmsByProjectIdAndDataType = function (project_id, task_type) {
        if (project_id) {
            $http({
                method: 'POST',
                dataType: 'json',
                url: urlsList.algorithm.load_manual_by_project,
                data: JSON.stringify({project_id: project_id, type: task_type}),
                contentType: 'application/json'
            }).then(function successCallback(response) {
                $scope.allAlgorithmsByProjectId = response.data;
                $scope.algorithmsByProjectAndCommon = $scope.allAlgorithmsByProjectId;
            }, function errorCallback(response) {
            })
        }
    };

    $scope.startProcessing = function (selectedObjects) {
        $scope.metrics = null;
        $scope.img = null;

        if (selectedObjects.selectedAlgorithm.id === undefined) {
            selectedObjects.selectedAlgorithm.id = -1;
        }

        $http({
            method: 'POST',
            dataType: 'json',
            url: urlsList.analysClassif.start_processing,
            data: JSON.stringify({
                "selectedProject": selectedObjects.selectedProject.id,
                "selectedRecord": selectedObjects.selectedRecord.id,
                "selectedAlgorithm": selectedObjects.selectedAlgorithm.id,
                "selectedResultType": selectedObjects.selectedResultType.id
            }),
            contentType: 'application/json'
        }).then(function successCallback(response) {
            var dataProcessingResult = response.data;
            $scope.type = dataProcessingResult['type'];
            switch ($scope.type) {
                case "train_save_metrics_graphics":
                    $scope.metrics = dataProcessingResult['metrics'];
                    $scope.img = dataProcessingResult['img'];
                    break;
                case "automaticle_best_model":
                    $scope.res = dataProcessingResult["res"];
                    if ("classification" in $scope.res){
                        $scope.bestClassif = $scope.res.classification.best;
                    }
                    if ("clustering" in $scope.res) {
                        $scope.bestCluster = $scope.res.clustering.best;
                    }

                    delete $scope.res.clustering.best;
                    delete $scope.res.classification.best;
                    break;
                default:
                    break;
            }
        }, function errorCallback(response) {
        })
    };

    $scope.loadAllAnalysClassifByProjectId = function (project_id) {
        if (project_id) {

        }
    };

    $scope.loadAllResultTypes = function (flagRemoveBestAlgType) {
        $http({
            method: 'GET',
            url: urlsList.analysClassif.load_all_result_types
        }).then(function successCallback(response) {
            $scope.resultTypes = response.data;
            if (flagRemoveBestAlgType) {
                // TODO
                // rewrite
                $scope.resultTypes.pop();
            }
        }, function errorCallback(response) {
        });
    };

    $scope.changeColor = function (styleVar) {
        switch (styleVar) {
            case "styleProject":
                $scope["styleRecord"]={color:'black'};
                $scope["styleAlgorithms"]={color:'black'};
                $scope["styleResultType"]={color:'black'};
                break;

            case "styleRecord":
                $scope["styleAlgorithms"]={color:'black'};
                $scope["styleResultType"]={color:'black'};
                break;

            case "styleAlgorithms":
                $scope["styleResultType"]={color:'black'};
                break;

            case "styleResultType":
                break;
        }

        $scope[styleVar]={color:'blue'};
    };

    //$scope.loadAllResultTypes();
});