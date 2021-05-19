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
"""TFX Custom Python Components."""


import sys
import os
import time
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

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from src.utils.ucaip_utils import AIPUtils


HYPERPARAM_FILENAME = 'hyperparameters.json'

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
def aip_model_uploader(
    project: Parameter[str],
    region: Parameter[str],
    model_display_name: Parameter[str],
    pushed_model_location: Parameter[str],
    serving_image_uri: Parameter[str],
    uploaded_model: OutputArtifact[UploadedModel]
):
    
    aip_utils = AIPUtils(project, region)
    
    pushed_model_dir = os.path.join(
        pushed_model_location, tf.io.gfile.listdir(pushed_model_location)[-1])
    
    logging.info(f"Model registry location: {pushed_model_dir}")
    
    response = aip_utils.upload_model(
        model_display_name=model_display_name,
        model_artifact_uri=pushed_model_dir,
        serving_image_uri=serving_image_uri
    )
    
    model_uri = response.result().model
    logging.info(f"Model uploaded to AI Platform: {model_uri}")
    uploaded_model.set_string_custom_property('model_uri', model_uri)