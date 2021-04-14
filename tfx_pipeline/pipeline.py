"""TFX training pipeline definition.
"""

import os
import sys
import logging

import tensorflow_model_analysis as tfma

import tfx
from tfx.proto import example_gen_pb2, transform_pb2
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
from ml_metadata.proto import metadata_store_pb2

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

from tfx_pipeline import config
from tfx_pipeline import components as custom_components
from model_src import features, runner
from utils import datasource_utils

RAW_SCHEMA_DIR = "model_src/raw_schema"
TRANSFORM_MODULE_FILE = "model_src/preprocessing.py"
TRAIN_MODULE_FILE = "model_src/runner.py"


def create_pipeline(
    metadata_connection_config: metadata_store_pb2.ConnectionConfig, 
    pipeline_root: str,
    num_epochs: data_types.RuntimeParameter,
    batch_size: data_types.RuntimeParameter,
    learning_rate: data_types.RuntimeParameter,
    hidden_units: data_types.RuntimeParameter,
):

    pipeline_components = []

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
    )
    pipeline_components.append(hyperparams_gen)
    
    # Get train source query.
    train_sql_query = datasource_utils.get_source_query(
        config.PROJECT, 
        config.REGION, 
        config.DATASET_DISPLAYNAME, 
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
        instance_name="TrainData",
    )
    pipeline_components.append(train_example_gen)
    
    # Get test source query.
    test_sql_query = datasource_utils.get_source_query(
        config.PROJECT, 
        config.REGION, 
        config.DATASET_DISPLAYNAME, 
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
        instance_name="TestData",
    )
    pipeline_components.append(test_example_gen)

    # Schema importer.
    schema_importer = ImporterNode(
        instance_name="Schema_Importer",
        source_uri=RAW_SCHEMA_DIR,
        artifact_type=tfx.types.standard_artifacts.Schema,
    )
    pipeline_components.append(schema_importer)

    # Statistics generation.
    statistics_gen = StatisticsGen(
        examples=train_example_gen.outputs.examples)
    pipeline_components.append(statistics_gen)
    
    # Example validation.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs.statistics,
        schema=schema_importer.outputs.result,
    )
    pipeline_components.append(example_validator)

    # Data transformation.
    transform = Transform(
        examples=train_example_gen.outputs.examples,
        schema=schema_importer.outputs.result,
        module_file=TRANSFORM_MODULE_FILE,
        splits_config=transform_pb2.SplitsConfig(
            analyze=['train'], transform=['train', 'eval'])
    )
    pipeline_components.append(transform)
    
    # Add dependency from example_validator to transform.
    transform.add_upstream_node(example_validator)

    # Custom model training.
    trainer = Trainer(
        custom_executor_spec=local_executor_spec
        if config.TRAINING_RUNNER == "local"
        else caip_executor_spec,
        module_file=TRAIN_MODULE_FILE,
        transformed_examples=transform.outputs.transformed_examples,
        schema=schema_importer.outputs.result,
        transform_graph=transform.outputs.transform_graph,
        train_args={'splits': ['train'], 'num_steps': num_epochs},
        eval_args={'splits': ['eval'], 'num_steps': None},
        hyperparameters=hyperparams_gen.outputs.hyperparameters,
        instance_name="CustomModel",
    )
    pipeline_components.append(trainer)
    
    # AutoML model training.
    automl_trainer = custom_components.automl_trainer(
        project=config.PROJECT,
        region=config.REGION,
        dataset_display_name=config.DATASET_DISPLAYNAME,
        model_display_name=config.AUTOML_MODEL_DISPLAYNAME,
        target_column=features.TARGET_FEATURE_NAME,
        data_split_column=config.DATA_SPLIT_COLUMN,
        exclude_cloumns=config.EXCLUDE_COLUMNS,
        schema=schema_importer.outputs.result,
    )
    pipeline_components.append(automl_trainer)

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
                                lower_bound={'value': float(config.ACCURACY_THRESHOLD)}))),
            ])
        ])

    # Custom model evaluation.
    evaluator = Evaluator(
        examples=test_example_gen.outputs.examples,
        example_splits=['test'],
        model=trainer.outputs.model,
        eval_config=eval_config,
        schema=schema_importer.outputs.result,
        instance_name="CustomModel",
    )
    pipeline_components.append(evaluator)
    
    # Get AutoML evaluation metrics.
    automl_metric_gen = custom_components.automl_metrics_gen(
        project=config.PROJECT,
        region=config.REGION,
        uploaded_model=automl_trainer.outputs.uploaded_model
    )
    pipeline_components.append(automl_metric_gen)
    
    # Validate custom model against AutoML model.
    validator = custom_components.custom_model_validator(
        model_evaluation=evaluator.outputs.evaluation,
        uploaded_model_evaluation=automl_metric_gen.outputs.evaluation,
    )
    pipeline_components.append(validator)

    exported_model_location = os.path.join(
        config.MODEL_REGISTRY_URI, config.DATASET_DISPLAYNAME)
    push_destination = tfx.proto.pusher_pb2.PushDestination(
        filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
            base_directory=exported_model_location
        )
    )
    
    # Push custom model to model registry.
    pusher = Pusher(
        model=trainer.outputs.model,
        model_blessing=validator.outputs.blessing,
        push_destination=push_destination,
    )
    pipeline_components.append(pusher)
    
    # Upload custom trained model to AI Platform.
    aip_model_uploader = custom_components.aip_model_uploader(
        project=config.PROJECT,
        region=config.REGION,
        model_display_name=config.CUSTOM_MODEL_DISPLAYNAME,
        pushed_model_location=exported_model_location,
        serving_image_uri=config.SERVING_IMAGE_URI,
    )
    pipeline_components.append(aip_model_uploader)
    
    # Add dependency from pusher to aip_model_uploader.
    aip_model_uploader.add_upstream_node(pusher)

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