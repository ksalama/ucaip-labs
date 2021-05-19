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
"""Define a utility class to encapsulate the uCAIP SDK.

This should be a seperate reusably library.
"""

from datetime import datetime
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.cloud.aiplatform import gapic as aip
from google.cloud import aiplatform_v1beta1 as aip_beta
import tensorflow.io as tf_io

DATASET_METADATA_SCHEMA_URI = "gs://google-cloud-aiplatform/schema/dataset/metadata/tabular_1.0.0.yaml"
TRAINING_TAKS_DEFINITION_URI = "gs://google-cloud-aiplatform/schema/trainingjob/definition/automl_tabular_1.0.0.yaml"

class AIPUtils:
    def __init__(self, project, region):
        self.project = project
        self.region = region
        self.parent = f"projects/{project}/locations/{region}"
        self.aip_endpoint = f"{region}-aiplatform.googleapis.com"
        self.client_options = {"api_endpoint": self.aip_endpoint}
        
        # Init the service clients
        self.dataset_client = aip.DatasetServiceClient(
            client_options=self.client_options)
        self.automl_pipeline_client = aip.PipelineServiceClient(
            client_options=self.client_options)
        self.model_client = aip.ModelServiceClient(
            client_options= self.client_options)
        self.job_client = aip.JobServiceClient(
            client_options=self.client_options)
        self.endpoint_client = aip.EndpointServiceClient(
            client_options=self.client_options)
        self.prediction_client = aip.PredictionServiceClient(
            client_options=self.client_options)
        self.prediction_client_beta = aip_beta.PredictionServiceClient(
            client_options=self.client_options)
        
        # Validate the uniqueness of the datasets display names.
        self.list_datasets()

        # Validate the uniqueness of the model display names.
        self.list_models()
            
        # Validate the uniqueness of the endpoint display names.    
        self.list_endpoints()
        
    def list_datasets(self):
        datasets = self.dataset_client.list_datasets(parent=self.parent)
        dataset_display_names = [
            dataset.display_name for dataset in datasets]
        assert len(dataset_display_names) == len(set(dataset_display_names)), "Dataset display names are not unique."
        return datasets
    
    
    def get_dataset_by_display_name(self, dataset_display_name: str):
        result = None
        
        datasets = self.list_datasets()
        for dataset in datasets:
            if dataset.display_name == dataset_display_name:
                result = dataset
                break
                
        return result
    
    def get_dataset_by_uri(self, dataset_uri: str):
        return self.dataset_client.get_dataset(name=dataset_uri)


    def create_dataset(self, dataset_display_name: str, metadata_spec: dict):
        if self.get_dataset_by_display_name(dataset_display_name):
            raise ValueError(
                f"Dataset with the Display Name {dataset_display_name} already exists.")
        
        metadata = json_format.ParseDict(metadata_spec, Value())
        dataset_desc = {
            "display_name": dataset_display_name,
            "metadata_schema_uri": DATASET_METADATA_SCHEMA_URI,
            "metadata": metadata,
        }
        
        return self.dataset_client.create_dataset(
            parent=self.parent, dataset=dataset_desc)
    
    
    def list_models(self):
        models = self.model_client.list_models(parent=self.parent)
        model_display_names = [
            model.display_name for model in models]
        assert len(model_display_names) == len(set(model_display_names)), "Model display names are not unique."
        return models
    
    
    def get_model_by_display_name(self, model_display_name: str):
        result = None
        for model in self.list_models():
            if model.display_name == model_display_name:
                result = model
                break
        return result
        
    
    
    def train_automl_table(self,
        dataset_display_name: str,
        model_display_name: str,
        training_task_inputs_spec: dict,
        fraction_split: dict=None,
        predefined_split: dict=None
                                        
    ):
        
        if self.get_model_by_display_name(model_display_name):
            raise ValueError(
                f"Model with the Display Name {model_display_name} already exists.")
            
        dataset = self.get_dataset_by_display_name(dataset_display_name)
        if not dataset:
            raise ValueError(
                f"Dataset with Display Name {dataset_display_name} does not exists.")
            
        dataset_id = dataset.name.split('/')[-1]
            
        training_task_inputs = json_format.ParseDict(training_task_inputs_spec, Value())
        train_display_name =  f"train_{model_display_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        training_pipeline = {
            "display_name": train_display_name,
            "training_task_definition": TRAINING_TAKS_DEFINITION_URI,
            "training_task_inputs": training_task_inputs,
            "input_data_config": {
                "dataset_id": dataset_id,
                "fraction_split": fraction_split,
                "predefined_split": predefined_split
            },
            "model_to_upload": {"display_name": model_display_name},
        }

        training_job = self.automl_pipeline_client.create_training_pipeline(
            parent=self.parent, training_pipeline=training_pipeline
        )
        return training_job
    
    
    def get_automl_training_job_by_uri(self, job_uri: str):
        return self.automl_pipeline_client.get_training_pipeline(name=job_uri)
    
    def get_custom_training_job_by_uri(self, job_uri: str):
        return self.job_client.get_custom_job(name=job_uri)
    
    def get_batch_prediction_job_by_uri(self, job_uri: str):
        return self.job_client.get_batch_prediction_job(name=job_uri)
    
    def get_evaluation_results_by_model_uri(self, model_uri: str):
        return self.model_client.list_model_evaluations(parent=model_uri)
    
    def get_evaluation_results_by_model_display_name(self, model_display_name: str):
        model = self.get_model_by_display_name(model_display_name)
        if not model:
            raise ValueError(
                f"Model with Display Name {model_display_name} does not exists.")
            
        return self.model_client.list_model_evaluations(parent=model.name)
    
    
    def submit_custom_job(
        self,
        model_display_name: str,
        training_spec: dict,
        training_dir: str
    ):
        
        if self.get_model_by_display_name(model_display_name):
            raise ValueError(
                f"Model with the Display Name {model_display_name} already exists.")
            
        job_display_name = f"train_{model_display_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        custom_job = {
            "display_name": job_display_name,
            "job_spec": {
                "worker_pool_specs": training_spec,
                "base_output_directory": {
                    "output_uri_prefix": training_dir
                }
            }
        }



        job = self.job_client.create_custom_job(
            parent=self.parent, custom_job=custom_job)
        return job
    
    
    def upload_model(
        self,
        model_display_name: str,
        model_artifact_uri: str,
        serving_image_uri: str,
        predict_schemata: dict = None
    ):
        
        if self.get_model_by_display_name(model_display_name):
            raise ValueError(
                f"Model with the Display Name {model_display_name} already exists.")
            
        model_spec = {
            "display_name": model_display_name,
            "artifact_uri": model_artifact_uri,
            "predict_schemata": predict_schemata,
            "container_spec": {
                "image_uri": serving_image_uri,
                "command": [],
                "args": [],
                "env": [{"name": "env_name", "value": "env_value"}],
                "ports": [{"container_port": 8080}],
                "predict_route": "",
                "health_route": "",
            },
        }
        
        response = self.model_client.upload_model(
            model=model_spec, parent=self.parent)
        return response
    
    
    def list_endpoints(self):
        endpoints = self.endpoint_client.list_endpoints(parent=self.parent)
        endpoint_display_names = [
            endpoint.display_name for endpoint in endpoints]
        assert len(endpoint_display_names) == len(set(endpoint_display_names)), "Endpoint display names are not unique."
        return endpoints
    
    
    def get_endpoint_by_display_name(self, endpoint_display_name: str):
        result = None
        endpoints = self.list_endpoints()
        for endpoint in endpoints:
            if endpoint.display_name == endpoint_display_name:
                result = endpoint
                break
        
        return result
    
    
    def create_endpoint(self, endpoint_display_name: str):
        
        if self.get_endpoint_by_display_name(endpoint_display_name):
            raise ValueError(
                f"Endpoint with the Display Name {endpoint_display_name} already exists.")
            
        response = self.endpoint_client.create_endpoint(
            parent=self.parent,
            endpoint=aip.Endpoint(display_name=endpoint_display_name)
        )
        
        return response
    
    
    def deploy_model(
        self,
        model_display_name: str,
        endpoint_display_name: str,
        dedicated_serving_resources_spec: dict,
        traffic_split: dict = {"0": 100},
        disable_container_logging: bool = False,
        enable_access_logging:bool = False,
        automatic_resources: bool = None,
    ):
        
        model = self.get_model_by_display_name(model_display_name)
        if not model:
            raise ValueError(
                f"Model with Display Name {model_display_name} does not exists.")
            
        endpoint = self.get_endpoint_by_display_name(endpoint_display_name)
        if not endpoint:
            raise ValueError(
                f"Endpoint with Display Name {endpoint_display_name} does not exists.")

        deployed_model = aip.DeployedModel(
            model=model.name,
            disable_container_logging=disable_container_logging,
            enable_access_logging=enable_access_logging,
            automatic_resources=None,
            dedicated_resources=dedicated_serving_resources_spec
        )

        response = self.endpoint_client.deploy_model(
            endpoint=endpoint.name,
            deployed_model=deployed_model,
            traffic_split=traffic_split
        )

        return response
    
    
    def predict_tabular_classifier(
        self,
        endpoint_uri: str,
        instance: dict
    ):
        instances = [json_format.ParseDict(instance, Value())]
        response = self.prediction_client.predict(
            endpoint=endpoint_uri, instances=instances#, parameters=parameters
        )
        return response
    
    
    def explain_tabular_classifier(
        self,
        endpoint_uri,
        instance
    ):
        instances = [json_format.ParseDict(instance, Value())]
        response = self.prediction_client_beta.explain(
            endpoint=endpoint_uri, 
            instances=instances
        )
        return response
    
    
    def submit_batch_prediction_job(
        self,
        model_display_name: str, 
        gcs_data_uri_pattern: str, 
        gcs_output_uri: str,
        dedicated_resources: dict,
        instances_format: str,
        predictions_format: str='jsonl'
        
    ):
        
        model = self.get_model_by_display_name(model_display_name)
        if not model:
            raise ValueError(
                f"Model with Display Name {model_display_name} does not exists.")
            
        serving_data_uris = tf_io.gfile.glob(gcs_data_uri_pattern)

        job_name = f"batch_predict_{model_display_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        batch_prediction_job = {
            "display_name": job_name,
            "model": model.name,
            "input_config": {
                "instances_format": instances_format,
                "gcs_source": {"uris": serving_data_uris},
            },
            "output_config": {
                "predictions_format": predictions_format,
                "gcs_destination": {"output_uri_prefix": gcs_output_uri},
            },
            "dedicated_resources": dedicated_resources
        }

        response = self.job_client.create_batch_prediction_job(
            parent=self.parent, batch_prediction_job=batch_prediction_job
        )
        return response
        
    
