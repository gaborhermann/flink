#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

angular.module('flinkApp', ['ui.router', 'angularMoment'])

# --------------------------------------

.run ($rootScope) ->
  $rootScope.sidebarVisible = false
  $rootScope.showSidebar = ->
    $rootScope.sidebarVisible = !$rootScope.sidebarVisible
    $rootScope.sidebarClass = 'force-show'

# --------------------------------------

.value 'flinkConfig', {
  "refresh-interval": 10000
}

# --------------------------------------

.run (JobsService, MainService, flinkConfig, $interval) ->
  MainService.loadConfig().then (config) ->
    angular.extend flinkConfig, config

    JobsService.listJobs()

    $interval ->
      JobsService.listJobs()
    , flinkConfig["refresh-interval"]


# --------------------------------------

.config ($uiViewScrollProvider) ->
  $uiViewScrollProvider.useAnchorScroll()

# --------------------------------------

.config ($stateProvider, $urlRouterProvider) ->
  $stateProvider.state "overview",
    url: "/overview"
    views:
      main:
        templateUrl: "partials/overview.html"
        controller: 'OverviewController'

  .state "running-jobs",
    url: "/running-jobs"
    views:
      main:
        templateUrl: "partials/jobs/running-jobs.html"
        controller: 'RunningJobsController'
  
  .state "completed-jobs",
    url: "/completed-jobs"
    views:
      main:
        templateUrl: "partials/jobs/completed-jobs.html"
        controller: 'CompletedJobsController'

  .state "single-job",
    url: "/jobs/{jobid}"
    abstract: true
    views:
      main:
        templateUrl: "partials/jobs/job.html"
        controller: 'SingleJobController'

  .state "single-job.plan",
    url: ""
    abstract: true
    views:
      details:
        templateUrl: "partials/jobs/job.plan.html"
        controller: 'JobPlanController'

  .state "single-job.plan.overview",
    url: ""
    views:
      'node-details':
        templateUrl: "partials/jobs/job.plan.node-list.overview.html"
        controller: 'JobPlanOverviewController' 

  .state "single-job.plan.accumulators",
    url: "/accumulators"
    views:
      'node-details':
        templateUrl: "partials/jobs/job.plan.node-list.accumulators.html"
        controller: 'JobPlanAccumulatorsController' 

  .state "single-job.timeline",
    url: "/timeline"
    views:
      details:
        templateUrl: "partials/jobs/job.timeline.html"

  .state "single-job.timeline.vertex",
    url: "/{vertexId}"
    views:
      vertex:
        templateUrl: "partials/jobs/job.timeline.vertex.html"
        controller: 'JobTimelineVertexController'

  .state "single-job.statistics",
    url: "/statistics"
    views:
      details:
        templateUrl: "partials/jobs/job.statistics.html"

  .state "single-job.exceptions",
    url: "/exceptions"
    views:
      details:
        templateUrl: "partials/jobs/job.exceptions.html"
        controller: 'JobExceptionsController'

  .state "single-job.properties",
    url: "/properties"
    views:
      details:
        templateUrl: "partials/jobs/job.properties.html"
        controller: 'JobPropertiesController'

  .state "single-job.config",
    url: "/config"
    views:
      details:
        templateUrl: "partials/jobs/job.config.html"

  .state "taskmanagers",
    url: "/taskmanagers"
    views:
      main:
        templateUrl: "partials/taskmanagers/index.html"
        controller: 'TaskManagersController'

  .state "jobmanager",
      url: "/jobmanager"
      views:
        main:
          templateUrl: "partials/jobmanager/index.html"

  .state "jobmanager.config",
    url: "/config"
    views:
      details:
        templateUrl: "partials/jobmanager/config.html"
        controller: 'JobManagerConfigController'

  .state "jobmanager.stdout",
    url: "/stdout"
    views:
      details:
        templateUrl: "partials/jobmanager/stdout.html"

  .state "jobmanager.logfile",
    url: "/logfile"
    views:
      details:
        templateUrl: "partials/jobmanager/logfile.html"

  $urlRouterProvider.otherwise "/overview"
