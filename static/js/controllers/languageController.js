myApp.controller("languageController", ['$scope', 'ModalService', '$http', function ($scope, ModalService, $http) {
    $scope.allLanguages = [{"title": "Русский"}];
    $scope.selectedLanguage = $scope.allLanguages[0];
}]);