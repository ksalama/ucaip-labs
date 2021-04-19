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
"""Define KubeflowDagRunner to run the training pipeline using KFP."""

import os
from absl import logging

import kfp
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2
from tfx.utils import telemetry_utils

try:
    from tfx_pipeline import config, pipeline
except:
    import config, pipeline

def run():
    """Define a kubeflow pipeline."""

    # Define pipeline runtime parameters.
    
    num_epochs_param = data_types.RuntimeParameter(
        name="num-epochs", default=5, ptype=int
    )
    
    batch_size_param = data_types.RuntimeParameter(
        name="batch-size", default=512, ptype=int
    )
    
    learning_rate_param = data_types.RuntimeParameter(
        name="learning-rate", default=0.0001, ptype=float
    )
    
    hidden_units_param = data_types.RuntimeParameter(
        name="hidden-units", default="64,64", ptype=str
    )
    
    # Create pipeline metadata store connection config.
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        pipeline_operator_funcs=kubeflow_dag_runner.get_default_pipeline_operator_funcs(
            config.USE_KFP_SA == "True"
        ),
        tfx_image=config.IMAGE_URI,
    )
    
    # Label the Kubeflow pods.
    pod_labels = kubeflow_dag_runner.get_default_pod_labels()
    pod_labels.update({telemetry_utils.LABEL_KFP_SDK_ENV: config.PIPELINE_NAME})
    
     # Prepare pipeline root.
    pipeline_root = os.path.join(
        config.ARTIFACT_STORE_URI,
        config.PIPELINE_NAME,
        kfp.dsl.RUN_ID_PLACEHOLDER,
    )
    
    # Create pipeline.
    kfp_pipeline = pipeline.create_pipeline(
        metadata_connection_config=metadata_config, 
        pipeline_root=pipeline_root,
        num_epochs=num_epochs_param,
        batch_size=batch_size_param,
        learning_rate=learning_rate_param,
        hidden_units=hidden_units_param   
    )
    
    # Create Kubeflow dag runner.
    kf_dag_runner = kubeflow_dag_runner.KubeflowDagRunner(
        config=runner_config, pod_labels_to_attach=pod_labels
    )
    
    # Run the pipeline.
    kf_dag_runner.run(kfp_pipeline)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()