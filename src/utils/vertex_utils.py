# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Define a utility class to encapsulate the MB SDK."""

import logging
import copy
from datetime import datetime

from google.protobuf.duration_pb2 import Duration
from google.cloud import aiplatform as vertex_ai
from google.cloud import aiplatform_v1beta1 as vertex_ai_beta


DEFAULT_CUSTOM_TRAINING_JOB_PREFIX = "custom-job"
DEFAULT_BATCH_PREDICTION_JOB_PREFIX = "prediction-job"


class VertexClient:
    def __init__(self, project, region, staging_bucket=None):

        self.project = project
        self.region = region
        self.staging_bucket = staging_bucket

        self.parent = f"projects/{project}/locations/{region}"
        self.client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}

        self.job_client_beta = vertex_ai_beta.JobServiceClient(
            client_options=self.client_options
        )
        self.tensorboard_client_beta = vertex_ai_beta.TensorboardServiceClient(
            client_options=self.client_options
        )

        vertex_ai.init(
            project=self.project,
            location=self.region,
            staging_bucket=self.staging_bucket,
        )

        # Validate the uniqueness of the datasets display names.
        self.list_datasets()

        # Validate the uniqueness of the model display names.
        self.list_models()

        # Validate the uniqueness of the endpoint display names.
        self.list_endpoints()

        # Validate the uniqueness of the tensorboard display names.
        self.list_tensorboard_instances()

        logging.info(
            f"Uniqueness of objects in project:{project} region:{region} validated."
        )
        logging.info(f"Vertex AI client initialized.")

    #####################################################################################
    # Dataset methods
    #####################################################################################

    def list_datasets(self):
        datasets = vertex_ai.TabularDataset.list()
        dataset_display_names = [dataset.display_name for dataset in datasets]
        assert len(dataset_display_names) == len(
            set(dataset_display_names)
        ), "Dataset display names are not unique."
        return datasets

    def get_dataset_by_display_name(self, display_name: str):
        dataset = None

        datasets = self.list_datasets()
        for entry in datasets:
            if entry.display_name == display_name:
                dataset = entry
                break

        return dataset

    def create_dataset_bq(self, display_name: str, bq_uri: str):
        if self.get_dataset_by_display_name(display_name):
            raise ValueError(
                f"Dataset with the Display Name {display_name} already exists."
            )

        return vertex_ai.TabularDataset.create(
            display_name=display_name, bq_source=bq_uri
        )

    #####################################################################################
    # ML metadata methods
    #####################################################################################

    def set_experiment(self, experiment_name):
        experiment_name = experiment_name.replace("_", "-")
        vertex_ai.init(
            project=self.project,
            location=self.region,
            staging_bucket=self.staging_bucket,
            experiment=experiment_name,
        )

    def start_experiment_run(self, run_name=None, experiment_name=None):
        if experiment_name:
            print("experiment_name is none!")
            self.set_experiment(experiment_name)
        if not run_name:
            run_name = f"run-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        run_name = run_name.replace("_", "-")
        vertex_ai.start_run(run_name)
        return run_name

    def get_experiment_df(self, experiment_name):
        return vertex_ai.get_experiment_df(experiment_name)

    def log_params(self, params):
        vertex_ai.log_params(params)

    def log_metrics(self, metrics):
        vertex_ai.log_metrics(metrics)

    #####################################################################################
    # Model training methods
    #####################################################################################

    def submit_custom_job(
        self,
        training_spec: dict,
        experiment_dir: str,
        job_display_name: str = None,
        service_account: str = None,
        tensorboard_resource_name: str = None,
    ):
        if not job_display_name:
            job_display_name = f"{DEFAULT_CUSTOM_TRAINING_JOB_PREFIX}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        custom_job = {
            "display_name": job_display_name,
            "job_spec": {
                "worker_pool_specs": training_spec,
                "base_output_directory": {"output_uri_prefix": experiment_dir},
                "service_account": service_account,
                "tensorboard": tensorboard_resource_name,
            },
        }

        job = self.job_client_beta.create_custom_job(
            parent=self.parent, custom_job=custom_job
        )
        return job

    def get_custom_job_by_uri(self, job_uri: str):
        return self.job_client_beta.get_custom_job(name=job_uri)

    #####################################################################################
    # Model methods
    #####################################################################################

    def list_models(self):
        models = vertex_ai.Model.list()
        model_display_names = [model.display_name for model in models]
        assert len(model_display_names) == len(
            set(model_display_names)
        ), "Model display names are not unique."
        return models

    def get_model_by_display_name(self, display_name: str):
        model = None
        for entry in self.list_models():
            if entry.display_name == display_name:
                model = entry
                break
        return model

    def upload_model(
        self,
        display_name: str,
        model_artifact_uri: str,
        serving_image_uri: str,
        instance_schema_uri: str = None,
        parameters_schema_uri: str = None,
    ):

        if self.get_model_by_display_name(display_name):
            raise ValueError(
                f"Model with the Display Name {display_name} already exists."
            )

        return vertex_ai.Model.upload(
            display_name=display_name,
            artifact_uri=model_artifact_uri,
            serving_container_image_uri=serving_image_uri,
            parameters_schema_uri=parameters_schema_uri,
            instance_schema_uri=instance_schema_uri,
        )

    #####################################################################################
    # Endpoint methods
    #####################################################################################

    def list_endpoints(self):
        endpoints = vertex_ai.Endpoint.list()
        endpoint_display_names = [endpoint.display_name for endpoint in endpoints]
        assert len(endpoint_display_names) == len(
            set(endpoint_display_names)
        ), "Endpoint display names are not unique."
        return endpoints

    def get_endpoint_by_display_name(self, endpoint_display_name: str):
        endpoint = None
        endpoints = self.list_endpoints()
        for entry in endpoints:
            if entry.display_name == endpoint_display_name:
                endpoint = entry
                break
        return endpoint

    def create_endpoint(self, display_name: str, raise_error_if_exists: bool = False):
        endpoint = self.get_endpoint_by_display_name(display_name)
        if endpoint:
            if raise_error_if_exists:
                raise ValueError(
                    f"Endpoint with the Display Name {display_name} already exists."
                )

        else:
            endpoint = vertex_ai.Endpoint.create(display_name)
        return endpoint

    def get_deployed_models(self, endpoint_display_name: str):
        endpoint = self.get_endpoint_by_display_name(endpoint_display_name)
        if not endpoint:
            raise ValueError(
                f"Endpoint with the Display Name {endpoint_display_name} does not exist."
            )

        return endpoint.list_models()

    def deploy_model(
        self,
        model_display_name: str,
        endpoint_display_name: str,
        serving_resources_spec: dict,
    ):

        model = self.get_model_by_display_name(model_display_name)
        if not model:
            raise ValueError(
                f"Model with Display Name {model_display_name} does not exists."
            )

        endpoint = self.get_endpoint_by_display_name(endpoint_display_name)
        if not endpoint:
            raise ValueError(
                f"Endpoint with Display Name {endpoint_display_name} does not exists."
            )

        endpoint.deploy(model=model, **serving_resources_spec)

        deployed_model = None
        for deployed_model in endpoint.list_models():
            if deployed_model.display_name == model_display_name:
                break

        return deployed_model

    #####################################################################################
    # Online prediction methods
    #####################################################################################

    def predict(self, endpoint_display_name: str, instances: list):
        endpoint = self.get_endpoint_by_display_name(endpoint_display_name)
        # ISSUE: endpoint.predict crashes without this line!
        endpoint = vertex_ai.Endpoint(endpoint.gca_resource.name)
        return endpoint.predict(instances)

    def explain(self, endpoint_display_name: str, instances: list):
        endpoint = self.get_endpoint_by_display_name(endpoint_display_name)
        return endpoint.explain(instances)

    #####################################################################################
    # Batch prediction methods
    #####################################################################################

    def submit_batch_prediction_job(
        self,
        model_display_name: str,
        gcs_source_pattern: str,
        gcs_destination_prefix: str,
        instances_format: str,
        predictions_format: str = "jsonl",
        job_display_name: str = None,
        other_configurations: dict = None,
    ):

        model = self.get_model_by_display_name(model_display_name)
        if not model:
            raise ValueError(
                f"Model with Display Name {model_display_name} does not exists."
            )

        if not job_display_name:
            job_display_name = f"{DEFAULT_BATCH_PREDICTION_JOB_PREFIX}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return vertex_ai.BatchPredictionJob.create(
            job_display_name=job_display_name,
            model_name=model.name,
            gcs_source=gcs_source_pattern,
            gcs_destination_prefix=gcs_destination_prefix,
            instances_format=instances_format,
            predictions_format=predictions_format,
            **other_configurations,
        )

    #####################################################################################
    # Monitoring methods
    #####################################################################################

    def create_monitoring_job(
        self,
        job_name: str,
        dataset_display_name: str,
        endpoint_display_name: str,
        target_field: str,
        log_sample_rate: float = 0.1,
        monitor_interval: int = 1440,
        skew_thresholds: dict = None,
        drift_thresholds: dict = None,
        predict_instance_schema_uri: str = "",
        analysis_instance_schema_uri: str = "",
        notify_emails: str = None,
    ):

        dataset = self.get_dataset_by_display_name(dataset_display_name)
        if not dataset:
            raise ValueError(f"Dataset {dataset_display_name} does not exist!")
        endpoint = self.get_endpoint_by_display_name(endpoint_display_name)
        if not endpoint:
            raise ValueError(f"Endpoint {endpoint_display_name} does not exist!")

        model_ids = [model.id for model in endpoint.list_models()]

        skew_thresholds = {
            feature: vertex_ai_beta.ThresholdConfig(value=float(value))
            for feature, value in skew_thresholds.items()
        }

        drift_thresholds = {
            feature: vertex_ai_beta.ThresholdConfig(value=float(value))
            for feature, value in drift_thresholds.items()
        }

        skew_config = vertex_ai_beta.ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig(
            skew_thresholds=skew_thresholds
        )
        drift_config = vertex_ai_beta.ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
            drift_thresholds=drift_thresholds
        )

        random_sampling = vertex_ai_beta.SamplingStrategy.RandomSampleConfig(
            sample_rate=log_sample_rate
        )
        sampling_config = vertex_ai_beta.SamplingStrategy(
            random_sample_config=random_sampling
        )

        duration = Duration(seconds=monitor_interval)
        schedule_config = vertex_ai_beta.ModelDeploymentMonitoringScheduleConfig(
            monitor_interval=duration
        )

        dataset = self.get_dataset_by_display_name(dataset_display_name)
        bq_source_uri = dataset.gca_resource.metadata["inputConfig"]["bigquerySource"][
            "uri"
        ]

        training_dataset = (
            vertex_ai_beta.ModelMonitoringObjectiveConfig.TrainingDataset(
                target_field=target_field
            )
        )
        training_dataset.bigquery_source = vertex_ai_beta.types.io.BigQuerySource(
            input_uri=bq_source_uri
        )
        objective_config = vertex_ai_beta.ModelMonitoringObjectiveConfig(
            training_dataset=training_dataset,
            training_prediction_skew_detection_config=skew_config,
            prediction_drift_detection_config=drift_config,
        )

        objective_template = vertex_ai_beta.ModelDeploymentMonitoringObjectiveConfig(
            objective_config=objective_config
        )

        deployment_objective_configs = []
        for model_id in model_ids:
            objective_config = copy.deepcopy(objective_template)
            objective_config.deployed_model_id = model_id
            deployment_objective_configs.append(objective_config)

        alerting_config = None

        if notify_emails:
            email_config = vertex_ai_beta.ModelMonitoringAlertConfig.EmailAlertConfig(
                user_emails=notify_emails
            )
            alerting_config = vertex_ai_beta.ModelMonitoringAlertConfig(
                email_alert_config=email_config
            )

        job = vertex_ai_beta.ModelDeploymentMonitoringJob(
            display_name=job_name,
            endpoint=endpoint.name,
            model_deployment_monitoring_objective_configs=deployment_objective_configs,
            logging_sampling_strategy=sampling_config,
            model_deployment_monitoring_schedule_config=schedule_config,
            model_monitoring_alert_config=alerting_config,
            predict_instance_schema_uri=predict_instance_schema_uri,
            analysis_instance_schema_uri=analysis_instance_schema_uri,
        )

        response = self.job_client_beta.create_model_deployment_monitoring_job(
            parent=self.parent, model_deployment_monitoring_job=job
        )
        return response

    def list_monitoring_jobs(self):
        return self.job_client_beta.list_model_deployment_monitoring_jobs(
            parent=self.parent
        )

    def get_monitoring_job_by_name(self, job_name):
        job = None
        for entry in self.list_monitoring_jobs():
            if entry.display_name == job_name:
                job = entry
                break
        return job

    def pause_monitoring_job(self, job_name):
        job = self.get_monitoring_job_by_name(job_name)
        if not job:
            raise ValueError(f"Monitoring job {job_name} does not exist!")
        return self.job_client_beta.pause_model_deployment_monitoring_job(
            name=job.name
        )

    def delete_monitoring_job(self, job_name):
        job = self.get_monitoring_job_by_name(job_name)
        if not job:
            raise ValueError(f"Monitoring job {job_name} does not exist!")
        return self.job_client_beta.delete_model_deployment_monitoring_job(
            name=job.name
        )

    #####################################################################################
    # TensorBoards methods
    #####################################################################################

    def list_tensorboard_instances(self):
        tensorboards = self.tensorboard_client_beta.list_tensorboards(
            parent=self.parent
        )
        tensorboard_display_names = [
            tensorboard.display_name for tensorboard in tensorboards
        ]
        assert len(tensorboard_display_names) == len(
            set(tensorboard_display_names)
        ), "TensorBoard display names are not unique."
        return tensorboards

    def get_tensorboard_by_display_name(self, display_name: str):
        tensorboard_instance = None

        tensorboard_instances = self.list_tensorboard_instances()
        for entry in tensorboard_instances:
            if entry.display_name == display_name:
                tensorboard_instance = entry
                break

        return tensorboard_instance
