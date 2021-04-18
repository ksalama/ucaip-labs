"""TFX Custom Python Components."""


import os
import json
import warnings
import logging
from datetime import datetime

import tfx
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
from tensorflow_transform.tf_metadata import schema_utils

from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import InputArtifact, OutputArtifact, Parameter
from tfx.types.standard_artifacts import Artifact, PushedModel, HyperParameters, Schema, ModelEvaluation, ModelBlessing
from tfx.types.experimental.simple_artifacts import File as UploadedModel, Metrics as UploadedModelEvaluation


from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.cloud.aiplatform import gapic as aip


HYPERPARAM_FILENAME = 'hyperparameters.json'
SCHEMA_FILENAME = 'schema.pbtxt'
EVALUATION_RESULTS_FILENAME = 'evaluation_results.json'


@component
def hyperparameters_gen(
    num_epochs: Parameter[int],
    batch_size: Parameter[int],
    learning_rate: Parameter[float],
    hidden_units: Parameter[str],
    hyperparameters: OutputArtifact[HyperParameters]
):
    
    hp_dict = dict()
    hp_dict['num_epochs'] = num_epochs
    hp_dict['batch_size'] = batch_size
    hp_dict['learning_rate'] = learning_rate
    hp_dict['hidden_units'] = [int(units) for units in hidden_units.split(',')]
    logging.info(f"Hyperparameters: {hp_dict}")
    
    hyperparams_uri = os.path.join(artifact_utils.get_single_uri([hyperparameters]), HYPERPARAM_FILENAME)
    io_utils.write_string_file(hyperparams_uri, json.dumps(hp_dict))
    logging.info(f"Hyperparameters are written to: {hyperparams_uri}")
    

    
@component
def automl_trainer(
    project: Parameter[str],
    region: Parameter[str],
    dataset_display_name: Parameter[str],
    model_display_name: Parameter[str],
    target_column: Parameter[str],
    data_split_column: Parameter[str],
    exclude_cloumns: Parameter[str],
    schema: InputArtifact[Schema],
    uploaded_model: OutputArtifact[UploadedModel]
):
    
    parent = f"projects/{project}/locations/{region}"
    api_endpoint = f"{region}-aiplatform.googleapis.com"
    client_options = {"api_endpoint": api_endpoint}

    logging.info(f"Retrieving {dataset_display_name} dataset id...")
    dataset_client = aip.DatasetServiceClient(client_options=client_options)
    for dataset in dataset_client.list_datasets(parent=parent):
        if dataset.display_name == dataset_display_name:
            dataset_uri = dataset.name
            break
    
    logging.info(f"{dataset_display_name} dataset uri: {dataset.name}")
    dataset_id = dataset.name.split('/')[-1]
    logging.info(f"{dataset_display_name} dataset id: {dataset_id}")

    schema_file = os.path.join(artifact_utils.get_single_uri([schema]), SCHEMA_FILENAME)
    logging.info(f"Loading schema from: {schema_file}")
    schema_obj = tfdv.load_schema_text(schema_file)
    raw_feature_spec = schema_utils.schema_as_feature_spec(schema_obj).feature_spec
    
    exclude_cloumns = exclude_cloumns.split(',')
    input_columns = [key for key in raw_feature_spec if key not in exclude_cloumns]
    
    transformations = [
        {"auto": {"column_name": column}} 
        for column in input_columns
    ]
    
    training_task_inputs_dict = {
        "targetColumn": target_column,
        "predictionType": "classification",
        "transformations": transformations,
        "trainBudgetMilliNodeHours": 1,
        "disableEarlyStopping": False,
        "optimizationObjective": "minimize-log-loss",
    }
    training_task_inputs = json_format.ParseDict(training_task_inputs_dict, Value())

    training_pipeline = {
        "display_name": f"train_{model_display_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "training_task_definition": "gs://google-cloud-aiplatform/schema/trainingjob/definition/automl_tabular_1.0.0.yaml",
        "training_task_inputs": training_task_inputs,
        "input_data_config": {
            "dataset_id": dataset_id,
            "predefined_split": {
               "key": data_split_column 
            }
        },
        "model_to_upload": {"display_name": model_display_name},
    }
    logging.info(f"AutoML Table training job specs: {training_pipeline}")

#     logging.info(f"Submitting AutoML Table training job...")
#     response = automl_pipeline_client.create_training_pipeline(
#         parent=parent, training_pipeline=training_pipeline
#     )
#     training_job = response.name
#     logging.info(f"Training job: {training_job}")
#     logging.info(f"Training job is running...")
    
#     response_result = response.result()
    
#     logging.info(f"AutoML training completed with status  {response_result.state}")
#     uploaded_model.set_string_custom_property('training_job', training_job)
    
    logging.info(f"Retrieving {model_display_name} model...")
    model_client = aip.ModelServiceClient(client_options=client_options)
    model_list = model_client.list_models(parent=parent)

    for entry in model_list:
        if entry.display_name == model_display_name:
            model_uri = entry.name
            break

    logging.info(f"Model uploaded to AI Platform: {model_uri}")
    uploaded_model.set_string_custom_property('model_uri', model_uri)
    
    

@component
def automl_metrics_gen(
    project: Parameter[str],
    region: Parameter[str],
    uploaded_model: InputArtifact[UploadedModel],
    evaluation: OutputArtifact[UploadedModelEvaluation]):
    
    parent = f"projects/{project}/locations/{region}"
    api_endpoint = f"{region}-aiplatform.googleapis.com"
    client_options = {"api_endpoint": api_endpoint}
    model_client = aip.ModelServiceClient(client_options=client_options)
    
    model_uri = uploaded_model.get_string_custom_property('model_uri')
    logging.info(f"Retrieving metrics for model: {model_uri}")
    
    model_evaluations = model_client.list_model_evaluations(parent=model_uri)
    metrics = list(model_evaluations)[0].metrics
    
    evaluation_results_dict = {
        'auPrc': metrics['auPrc'],
        'auRoc': metrics['auRoc'],
    }
    
    confusion_matrix_entries = metrics['confusionMatrix']['rows']
    
    tn = confusion_matrix_entries[0][0]
    fp = confusion_matrix_entries[0][1]
    fn = confusion_matrix_entries[1][0]
    tp = confusion_matrix_entries[1][1]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    evaluation_results_dict['accuracy'] = accuracy
    
    confidence_metrics = metrics['confidenceMetrics']
    
    f1score = None
    for m in metrics['confidenceMetrics']:
        entry = dict(m)
        f1Score = entry['f1Score']
        if 'confidenceThreshold' in entry and entry['confidenceThreshold'] >= 0.5:
            f1score = entry['f1Score']
            break
    evaluation_results_dict['f1Score'] = f1score
    
    logging.info(f"Evaluation results: {evaluation_results_dict}")
    
    evaluation_uri = os.path.join(artifact_utils.get_single_uri([evaluation]), EVALUATION_RESULTS_FILENAME)
    io_utils.write_string_file(evaluation_uri, json.dumps(evaluation_results_dict))
    logging.info(f"Evaluation results are written to: {evaluation_uri}")
    
    

@component
def custom_model_validator(
    model_evaluation: InputArtifact[ModelEvaluation],
    uploaded_model_evaluation: InputArtifact[UploadedModelEvaluation],
    blessing: OutputArtifact[ModelBlessing]
):
    model_accuracy = None
    uoloaded_model_accuracy = None
    
    model_evaluation_dir = artifact_utils.get_single_uri([model_evaluation])
    
    logging.info(f"Loading model evaluation from: {model_evaluation_dir}")
    model_metrics = list(tfma.load_metrics(model_evaluation_dir))[0].metric_keys_and_values
    logging.info(f"Model evaluation metrics: {model_metrics}")
    for entry in model_metrics:
        if entry.key.name == 'accuracy':
            model_accuracy = entry.value.double_value.value
            
    
    uploaded_model_evaluation_dir = artifact_utils.get_single_uri([uploaded_model_evaluation])
    logging.info(f"Loading uploaded model evaluation from: {uploaded_model_evaluation_dir}")
    
    uploaded_model_metrics = json.loads(
        io_utils.read_string_file(
            os.path.join(uploaded_model_evaluation_dir, EVALUATION_RESULTS_FILENAME)))
    logging.info(f'Uploaded model metric: {uploaded_model_metrics}')
    
    uoloaded_model_accuracy = uploaded_model_metrics['accuracy']
    
    blessed = model_accuracy > uoloaded_model_accuracy
    filename = 'BLESSED' if blessed else 'NOT_BLESSED'
    io_utils.write_string_file(
        os.path.join(blessing.uri, filename), '')
    blessing.set_int_custom_property('blessed', int(blessed))
    logging.info(f'Blessing result {blessed} written to {blessing.uri}')
        
    

@component
def aip_model_uploader(
    project: Parameter[str],
    region: Parameter[str],
    model_display_name: Parameter[str],
    pushed_model_location: Parameter[str],
    serving_image_uri: Parameter[str],
    uploaded_model: OutputArtifact[UploadedModel]
):
    
    parent = f"projects/{project}/locations/{region}"
    api_endpoint = f"{region}-aiplatform.googleapis.com"
    client_options = {"api_endpoint": api_endpoint}
    
    pushed_model_dir = os.path.join(
        pushed_model_location, tf.io.gfile.listdir(pushed_model_location)[-1])
    
    logging.info(f"Model registry location: {pushed_model_dir}")
    
    model_desc = {
        "display_name": model_display_name,
        "artifact_uri": pushed_model_dir,
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
    
    model_client = aip.ModelServiceClient(client_options=client_options)

    response = model_client.upload_model(
        model=model_desc,
        parent=parent
    )
    aip_model_uri = response.result().model
    logging.info(f"Model uploaded to AI Platform: {aip_model_uri}")
    uploaded_model.set_string_custom_property('model_uri', aip_model_uri)