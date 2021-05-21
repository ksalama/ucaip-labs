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
"""TFX pipeline definition."""

import os
import sys
import logging

import tensorflow_model_analysis as tfma

import tfx
from tfx.proto import example_gen_pb2, transform_pb2, trainer_pb2
from tfx.orchestration import pipeline, data_types
from tfx.dsl.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.extensions.google_cloud_ai_platform.trainer import (
    executor as ai_platform_trainer_executor,
)
from tfx.extensions.google_cloud_big_query.example_gen.component import (
    BigQueryExampleGen,
)
from tfx.components import StatisticsGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher
from tfx.components.common_nodes.importer_node import ImporterNode
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.dsl.experimental import latest_blessed_model_resolver

from ml_metadata.proto import metadata_store_pb2

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from src.pipelines import config
from src.pipelines import components as custom_components
from src.common import features
from src.model_training import runner
from src.utils import datasource_utils

RAW_SCHEMA_DIR = "src/raw_schema"
TRANSFORM_MODULE_FILE = "src/preprocessing/transformations.py"
TRAIN_MODULE_FILE = "src/model_training/runner.py"


def create_pipeline(
    metadata_connection_config: metadata_store_pb2.ConnectionConfig, 
    pipeline_root: str,
    num_epochs: data_types.RuntimeParameter,
    batch_size: data_types.RuntimeParameter,
    learning_rate: data_types.RuntimeParameter,
    hidden_units: data_types.RuntimeParameter,
):

    local_executor_spec = executor_spec.ExecutorClassSpec(
        trainer_executor.GenericExecutor
    )

    caip_executor_spec = executor_spec.ExecutorClassSpec(
        ai_platform_trainer_executor.GenericExecutor
    )
    
    # Hyperparameter generation.
    hyperparams_gen = custom_components.hyperparameters_gen(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
    ).with_id("HyperparamsGen")
    
    # Get train source query.
    train_sql_query = datasource_utils.get_training_source_query(
        config.PROJECT, 
        config.REGION, 
        config.DATASET_DISPLAY_NAME, 
        data_split='UNASSIGNED',
        limit=int(config.TRAIN_LIMIT)
    )

    train_output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(
                    name="train", hash_buckets=int(config.NUM_TRAIN_SPLITS)),
                example_gen_pb2.SplitConfig.Split(
                    name="eval", hash_buckets=int(config.NUM_EVAL_SPLITS)),
            ]
        )
    )
    
    # Train example generation.
    train_example_gen = BigQueryExampleGen(
        query=train_sql_query, 
        output_config=train_output_config,
    ).with_id("TrainDataGen")
    
    # Get test source query.
    test_sql_query = datasource_utils.get_training_source_query(
        config.PROJECT, 
        config.REGION, 
        config.DATASET_DISPLAY_NAME, 
        data_split='TEST',
        limit=int(config.TEST_LIMIT)
    )

    test_output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="test", hash_buckets=1),
            ]
        )
    )
    
    # Test example generation.
    test_example_gen = BigQueryExampleGen(
        query=test_sql_query, 
        output_config=test_output_config,
    ).with_id("TestDataGen")

    # Schema importer.
    schema_importer = ImporterNode(
        source_uri=RAW_SCHEMA_DIR,
        artifact_type=tfx.types.standard_artifacts.Schema,
    ).with_id("SchemaImporter")

    # Statistics generation.
    statistics_gen = StatisticsGen(
        examples=train_example_gen.outputs.examples
    ).with_id("StatisticsGen")
    
    # Example validation.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs.statistics,
        schema=schema_importer.outputs.result,
    ).with_id("ExampleValidator")

    # Data transformation.
    transform = Transform(
        examples=train_example_gen.outputs.examples,
        schema=schema_importer.outputs.result,
        module_file=TRANSFORM_MODULE_FILE,
        splits_config=transform_pb2.SplitsConfig(
            analyze=['train'], transform=['train', 'eval'])
    ).with_id("DataTransformer")
    
    # Add dependency from example_validator to transform.
    transform.add_upstream_node(example_validator)
    
    # Get the latest model to warmstart
    warmstart_model_resolver = Resolver(
        strategy_class=latest_artifacts_resolver.LatestArtifactsResolver,
        latest_model=tfx.types.Channel(type=tfx.types.standard_artifacts.Model)
    ).with_id("WarmstartModelResolver")


    # Model training.
    trainer = Trainer(
        custom_executor_spec=local_executor_spec
        if config.TRAINING_RUNNER == "local"
        else caip_executor_spec,
        module_file=TRAIN_MODULE_FILE,
        transformed_examples=transform.outputs.transformed_examples,
        schema=schema_importer.outputs.result,
        base_model=warmstart_model_resolver.outputs.latest_model,
        transform_graph=transform.outputs.transform_graph,
        train_args=trainer_pb2.TrainArgs(num_steps=0),
        eval_args=trainer_pb2.EvalArgs(num_steps=None),
        hyperparameters=hyperparams_gen.outputs.hyperparameters,
    ).with_id("ModelTrainer")
    
    # Get the latest blessed model (baseline) for model validation.
    baseline_model_resolver = Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=tfx.types.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.types.Channel(type=tfx.types.standard_artifacts.ModelBlessing),
    ).with_id("BaselineModelResolver")

    # Prepare evaluation config.
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name='serving_tf_example',
                label_key=features.TARGET_FEATURE_NAME,
                prediction_key='probabilities')
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[   
                    tfma.MetricConfig(class_name='ExampleCount'),
                    tfma.MetricConfig(
                        class_name='BinaryAccuracy',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': 0.8}),
                            # Change threshold will be ignored if there is no
                            # baseline model resolved from MLMD (first run).
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={'value': -1e-10}))),
            ])
        ])

    # Model evaluation.
    evaluator = Evaluator(
        examples=test_example_gen.outputs.examples,
        example_splits=['test'],
        model=trainer.outputs.model,
         baseline_model=baseline_model_resolver.outputs.model,
        eval_config=eval_config,
        schema=schema_importer.outputs.result,
    ).with_id("ModelEvaluator")

    exported_model_location = os.path.join(
        config.MODEL_REGISTRY_URI, config.DATASET_DISPLAY_NAME)
    push_destination = tfx.proto.pusher_pb2.PushDestination(
        filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
            base_directory=exported_model_location
        )
    )
    
    # Push custom model to model registry.
    pusher = Pusher(
        model=trainer.outputs.model,
        model_blessing=evaluator.outputs.blessing,
        push_destination=push_destination,
    ).with_id("ModelPusher")
    
    # Upload custom trained model to AI Platform.
    aip_model_uploader = custom_components.aip_model_uploader(
        project=config.PROJECT,
        region=config.REGION,
        model_display_name=config.MODEL_DISPLAY_NAME,
        pushed_model_location=exported_model_location,
        serving_image_uri=config.SERVING_IMAGE_URI,
    ).with_id("VertexUploader")

    # Add dependency from pusher to aip_model_uploader.
    aip_model_uploader.add_upstream_node(pusher)
    
    pipeline_components = [
        hyperparams_gen,
        train_example_gen,
        test_example_gen,
        statistics_gen,
        schema_importer,
        example_validator,
        transform,
        warmstart_model_resolver,
        trainer,
        baseline_model_resolver,
        evaluator,
        pusher,
        aip_model_uploader
    ]
    

    logging.info(
        f"Pipeline components: {[component.id for component in pipeline_components]}"
    )

    beam_pipeline_args = config.BEAM_DIRECT_PIPELINE_ARGS
    if config.BEAM_RUNNER == "DataflowRunner":
        beam_pipeline_args = config.BEAM_DATAFLOW_PIPELINE_ARGS

    logging.info(f"Beam pipeline args: {beam_pipeline_args}")

    return pipeline.Pipeline(
        pipeline_name=config.PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=pipeline_components,
        beam_pipeline_args=beam_pipeline_args,
        metadata_connection_config=metadata_connection_config,
        enable_cache=bool(config.ENABLE_CACHE),
    )