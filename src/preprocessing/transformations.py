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
"""TensorFlow Transform preprocessing function."""

import tensorflow as tf
import tensorflow_transform as tft

from src.common import features
from src.model_training import features as feature_info


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
      Map from string feature key to transformed feature operations.
    """

    outputs = {}

    for key in feature_info.FEATURE_NAMES:
        if key in feature_info.NUMERICAL_FEATURE_NAMES:
            outputs[features.transformed_name(key)] = tft.scale_to_z_score(inputs[key])

        elif key in features.categorical_feature_names():
            outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
                inputs[key],
                num_oov_buckets=1,
                vocab_filename=key,
            )

    outputs[feature_info.TARGET_FEATURE_NAME] = inputs[feature_info.TARGET_FEATURE_NAME]

    for key in outputs:
        outputs[key] = tf.squeeze(outputs[key], -1)

    return outputs
