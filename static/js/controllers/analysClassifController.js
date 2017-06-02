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

    //$scope.concatAlgorithmsTypes = function () {
    //    $scope.algorithmsByProjectAndCommon = $scope.allAlgorithmsByProjectId.concat($scope.commonAlgorithms);
    //};

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

                //console.log($scope.allAlgorithmsByProjectId);
                $scope.algorithmsByProjectAndCommon = $scope.allAlgorithmsByProjectId;
                //$scope.concatAlgorithmsTypes();
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
            console.log(dataProcessingResult);
            $scope.type = dataProcessingResult['type'];
            switch ($scope.type) {
                case "train_save_metrics_graphics":
                    $scope.metrics = dataProcessingResult['metrics'];
                    $scope.img = dataProcessingResult['img'];
                    break;
                case "automaticle_best_model":
                    $scope.res = dataProcessingResult["res"];
                    break;
                default:
                    break;
            }
        }, function errorCallback(response) {
        })
    };

    // $scope.loadAllCommonAlgorithms = function (project_id) {
    //     $http({
    //         method: 'GET',
    //         url: urlsList.algorithm.load_all_common
    //     }).then(function successCallback(response) {
    //         $scope.commonAlgorithms = response.data;
    //     }, function errorCallback(response) {
    //     });
    // };

    $scope.loadAllAnalysClassifByProjectId = function (project_id) {
        if (project_id) {

        }
    };

    $scope.loadAllResultTypes = function () {
        $http({
            method: 'GET',
            url: urlsList.analysClassif.load_all_result_types
        }).then(function successCallback(response) {
            $scope.resultTypes = response.data;
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

    //$scope.loadAllCommonAlgorithms();

    $scope.loadAllResultTypes();
});