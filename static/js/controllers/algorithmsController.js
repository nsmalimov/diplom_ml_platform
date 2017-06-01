myApp.controller("algorithmsController", ['$scope', 'ModalService', '$http', function ($scope, ModalService, $http) {
    $scope.allProjects = null;

    $scope.allAlgorithms = null;

    $scope.selectedAlgorithm = null;

    //$scope.selectedProject = null;

    $scope.allAlgorithmsByProjectId = null;

    $scope.project_id = null;

    $scope.fd = new FormData();

    $scope.commonAlgorithmsFlag = false;

    $scope.commonAlgorithms = null;

    $scope.commonAlgorithmsTitles = [];

    $scope.strButtonCheckCommonAlgorithmsFlag = "Показать стандартные";

    $scope.allTaskTypes = [{title: "Классификация", name: "classification"},
        {title: "Кластеризация", name: "clustering"}];

    $scope.selectedTaskType = null;

    $scope.projectObject = {
        title: null,
        description: null,
        type: null
    };

    $scope.reset = function () {
        $scope.selectedProject = null;

        $scope.selectedAlgorithm = null;

        $scope.allAlgorithmsByProjectId = null;

        $scope.project_id = null;

        $scope.fd = new FormData();

        $scope.commonAlgorithmsFlag = false;

        $scope.commonAlgorithms = null;

        $scope.commonAlgorithmsTitles = [];

        $scope.strButtonCheckCommonAlgorithmsFlag = "Показать стандартные";

        $scope.commonAlgorithmsTitles = [];

        $scope.projectObject = {
            title: null,
            description: null,
            type: null
        };
    };

    $scope.readFile = function (event) {
        var files = event.files;
        $scope.fd.append("file", files[0]);
    };

    $scope.uploadFile = function () {
        $scope.fd.append("title", $scope.projectObject.title);
        $scope.fd.append("description", $scope.projectObject.description);
        $scope.fd.append("type", $scope.projectObject.type);

        $http.post(baseUrl + urlsList.algorithm.upload_one + "/" + $scope.project_id, $scope.fd, {
            withCredentials: true,
            headers: {'Content-Type': undefined},
            transformRequest: angular.identity
        }).then(function (response) {
                $scope.loadAllAlgorithmsByProjectId($scope.project_id);
                document.getElementById('file-input').value = null;
            },
            function (response) {
            });
    };

    $scope.loadAllProjects = function () {
        $http({
            method: 'GET',
            url: urlsList.project.load_all
        }).then(function successCallback(response) {
            $scope.allProjects = response.data;
        }, function errorCallback(response) {
        });
    };

    $scope.loadAllAlgorithmsByProjectId = function (project_id) {
        $http({
            method: 'POST',
            dataType: 'json',
            url: urlsList.algorithm.load_manual_by_project,
            data: JSON.stringify({project_id: project_id, type: $scope.selectedTaskType.name}),
            contentType: 'application/json'
        }).then(function successCallback(response) {
            $scope.allAlgorithmsByProjectId = response.data;
            console.log($scope.allAlgorithmsByProjectId);
        }, function errorCallback(response) {
        })
    };

    $scope.loadAllCommonAlgorithms = function () {
        $http({
            method: 'POST',
            dataType: 'json',
            url: urlsList.algorithm.load_all_common,
            data: JSON.stringify({type: $scope.selectedTaskType.name}),
            contentType: 'application/json'
        }).then(function successCallback(response) {
            $scope.commonAlgorithms = response.data;
        }, function errorCallback(response) {
        });
    };

    $scope.loadAllProjects();

    $scope.taskTypeChange = function (selectedTaskType) {
        //$scope.reset();
        $scope.selectedTaskType = selectedTaskType;
        $scope.loadAllCommonAlgorithms();

        if ($scope.project_id) {
            $scope.loadAllAlgorithmsByProjectId($scope.project_id);
        }
    };

    $scope.onSelectUiClick = function (project_id) {
        $scope.allAlgorithmsByProjectId = [];
        $scope.project_id = project_id;
        $scope.loadAllAlgorithmsByProjectId(project_id);
    };

    $scope.showDescriptionModal = function () {
        ModalService.showModal({
            templateUrl: $scope.selectedTaskType.name === "classification" ? "/static/partials/modals/algorithmClassifDesc.html" : "/static/partials/modals/algorithmClusterDesc.html",
            controller: "modalController"
        }).then(function (modal) {
            modal.element.modal();
            modal.close.then(function () {
            });
        });
    };

    $scope.deleteObject = function (item) {
        item.object_type = "algorithm";
        $http({
            method: 'POST',
            dataType: 'json',
            url: urlsList.algorithm.delete,
            data: JSON.stringify(item),
            contentType: 'application/json'
        }).then(function successCallback(response) {
            $scope.loadAllAlgorithmsByProjectId(item.project_id);
            $scope.toggle();
        }, function errorCallback(response) {
        })
    };

    $scope.showCommonAlgorithms = function () {
        $scope.commonAlgorithmsFlag = !$scope.commonAlgorithmsFlag;
        if ($scope.strButtonCheckCommonAlgorithmsFlag === "Выбрать проект")
        {
            $scope.strButtonCheckCommonAlgorithmsFlag = "Показать стандартные";
        }
        else
        {
            $scope.strButtonCheckCommonAlgorithmsFlag = "Выбрать проект";
        }

    };

    $scope.checked = false;
    $scope.size = '100px';

    //Expression 'checked == true' in attribute 'psOpen'
    $scope.toggle = function (item) {
        if ($scope.selectedItem === item || $scope.selectedItem === undefined) {
            $scope.checked = !$scope.checked;
        }
        else {
            $scope.checked = true;
        }

        $scope.selectedItem = item;
    };

    $scope.mockRouteChange = function () {
        $scope.$broadcast('$locationChangeStart');
    };

    $scope.onopen = function () {
        alert('Open');
        console.log(this, $scope);
    };

    $scope.onclose = function () {
        alert('Close');
        console.log($scope);
    };
}]);