myApp.controller("mainController", function ($scope, dataService) {

    $scope.langName = dataService.getProperty();

    if ($scope.langName === "rus") {
        $scope.lang = rus;
    }
    if ($scope.langName === "eng") {
        $scope.lang = eng;
    }
    if ($scope.langName === "ch") {
        $scope.lang = ch;
    }

    $scope.sampleLang = rus;

}).config(function ($interpolateProvider) {
    $interpolateProvider.startSymbol('||').endSymbol('||');
});