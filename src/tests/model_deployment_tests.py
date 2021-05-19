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
"""Test an uploaded model to AI Platform."""

import os
import pytest
import logging
import tensorflow as tf

test_instance = {
    "dropoff_grid": tf.constant([["POINT(-87.6 41.9)"]]),
    "euclidean": tf.constant([[2064.2696]]),
    "loc_cross": tf.constant([[""]]),
    "payment_type": tf.constant([["Credit Card"]]),
    "pickup_grid": tf.constant([["POINT(-87.6 41.9)"]]),
    "trip_miles": tf.constant([[1.37]]),
    "trip_day": tf.constant([[12]], tf.dtypes.int64),
    "trip_hour": tf.constant([[16]], tf.dtypes.int64),
    "trip_month": tf.constant([[2]], tf.dtypes.int64),
    "trip_day_of_week": tf.constant([[4]], tf.dtypes.int64),
    "trip_seconds": tf.constant([[555]], tf.dtypes.int64),
}

SERVING_DEFAULT_SIGNATURE_NAME = 'serving_default'

from src.utils.ucaip_utils import AIPUtils

def test_model_artifact():
    
    project = os.getenv("PROJECT")
    region = os.getenv("REGION")
    model_display_name = os.getenv("MODEL_DISPLAY_NAME")
    
    
    aip_utils = AIPUtils(project, region)
    model_desc = aip_utils.get_model_by_display_name(model_display_name)
    artifact_uri = model_desc.artifact_uri
    logging.info(f"Model artifact uri:{artifact_uri}")
    assert tf.io.gfile.exists(artifact_uri), f"Model artifact uri {artifact_uri} does not exist!"
    
    saved_model_dir = os.path.join(artifact_uri, tf.io.gfile.listdir(artifact_uri)[-1])
    saved_model = tf.saved_model.load(saved_model_dir)
    logging.info("Model loaded successfully.")
    
    assert SERVING_DEFAULT_SIGNATURE_NAME in saved_model.signatures, f"{SERVING_DEFAULT_SIGNATURE_NAME} not in model signatures!"
    
    prediction_fn = saved_model.signatures['serving_default']
    predictions = prediction_fn(**test_instance)
    logging.info("Model produced predictions.")
    
    keys = ['classes', 'scores']
    for key in keys:
        assert key in predictions, f"{key} in prediction outputs!"
    
    assert predictions['classes'].shape == (1, 2), f"Invalid output classes shape: {predictions['classes'].shape}!"
    assert predictions['scores'].shape == (1, 2), f"Invalid output scores shape: {predictions['scores'].shape}!"