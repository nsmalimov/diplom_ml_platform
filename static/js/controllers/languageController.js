myApp.controller("languageController", ['$scope', 'ModalService', '$http', 'dataService',
    function ($scope, ModalService, $http, dataService) {
    $scope.allLanguages = [{"title": "Русский", "name": "rus"},
        {"title": "English", "name": "eng"},
        {"title": "中國", "name": "ch"}];

    $scope.selectedLanguage = $scope.allLanguages[0];



    dataService.setProperty($scope.selectedLanguage.name);
}]);