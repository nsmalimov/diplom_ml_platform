var myApp = angular.module('myApp', ['ng.bs.dropdown', 'pageslide-directive', 'ngSanitize', 'ng-showdown', 'ui.bootstrap', 'ngRoute', 'ui.select', 'angularModalService', 'ngAnimate']).config(function ($routeProvider, $locationProvider) {
    $routeProvider.when('/',
        {
            templateUrl: '/partials/main.html'
            //controller:'QuestionController'
        });
    $routeProvider.when('/projects',
        {
            templateUrl: '/partials/projects.html',
            controller: 'projectController'
        });
    $routeProvider.when('/analysclassif',
        {
            templateUrl: '/partials/analysClassif.html',
            controller:'analysClassifController'
        });
    $routeProvider.when('/data',
        {
            templateUrl: '/partials/data.html',
            controller:'dataController'
        });
    $routeProvider.when('/algorithms',
        {
            templateUrl: '/partials/algorithms.html',
            controller:'algorithmsController'
        });
    $locationProvider.hashPrefix('');
});