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
"""TFX pipeline configurations."""

import os

DATASET_DISPLAY_NAME = os.getenv("DATASET_DISPLAY_NAME", "chicago_taxi_tips")
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", f"{DATASET_DISPLAY_NAME}_classifier_custom")

DATA_SPLIT_COLUMN = 'data_split'
EXCLUDE_COLUMNS = ','.join(['trip_start_timestamp'])
TRAIN_LIMIT = os.getenv("TRAIN_LIMIT", "0")
TEST_LIMIT = os.getenv("TEST_LIMIT", "0")

PIPELINE_NAME = os.getenv("PIPELINE_NAME", f"{DATASET_DISPLAY_NAME}_train_pipeline")
PIPELINE_NAME = PIPELINE_NAME.replace('_', '-')

PROJECT = os.getenv("PROJECT", "ksalama-cloudml")
REGION = os.getenv("REGION", "us-central1")

GCS_LOCATION = os.getenv(
    "GCS_LOCATION", "gs://ksalama-cloudml-us/ucaip_demo/chicago_taxi"
)

ARTIFACT_STORE_URI = os.path.join(GCS_LOCATION, "tfx_arficats")
MODEL_REGISTRY_URI = os.getenv(
    "MODEL_REGISTRY_URI",
    os.path.join(GCS_LOCATION, "model_registry"),
)

NUM_TRAIN_SPLITS = os.getenv("NUM_TRAIN_SPLITS", "4")
NUM_EVAL_SPLITS = os.getenv("NUM_EVAL_SPLITS", "1")
ACCURACY_THRESHOLD = os.getenv("ACCURACY_THRESHOLD", "0.85")

USE_KFP_SA = os.getenv("USE_KFP_SA", "False")

IMAGE_URI = os.getenv(
    "IMAGE_URI", f"gcr.io/{PROJECT}/tfx_{DATASET_DISPLAY_NAME}:latest"
)

BEAM_RUNNER = os.getenv("BEAM_RUNNER", "DirectRunner")
BEAM_DIRECT_PIPELINE_ARGS = [
    f"--project={PROJECT}",
    f"--temp_location={os.path.join(GCS_LOCATION, 'temp')}",
]
BEAM_DATAFLOW_PIPELINE_ARGS = [
    f"--project={PROJECT}",
    f"--temp_location={os.path.join(GCS_LOCATION, 'temp')}",
    f"--region={REGION}",
    "--runner=DataflowRunner",
]


TRAINING_RUNNER = os.getenv("TRAINING_RUNNER", "local")
AI_PLATFORM_TRAINING_ARGS = {
    "project": PROJECT,
    "region": REGION,
    "masterConfig": {"imageUri": IMAGE_URI},
}


SERVING_RUNTIME = os.getenv("SERVING_RUNTIME", 'tf2-cpu.2-3')
SERVING_IMAGE_URI = f"gcr.io/cloud-aiplatform/prediction/{SERVING_RUNTIME}:latest"

ENABLE_CACHE = os.getenv("ENABLE_CACHE", "FALSE")


os.environ['PROJECT'] = PROJECT
os.environ['PIPELINE_NAME'] = PIPELINE_NAME
os.environ['IMAGE_URI'] = IMAGE_URI