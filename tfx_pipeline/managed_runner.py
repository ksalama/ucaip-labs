import os
from absl import logging
from aiplatform.pipelines import client
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner

try:
    from tfx_pipeline import config, pipeline
    from model_src import defaults
except:
    import config, pipeline
    


def compile_pipeline():
    
    pipeline_root = os.path.join(
        config.ARTIFACT_STORE_URI,
        config.PIPELINE_NAME,
    )
    
    managed_pipeline = pipeline.create_pipeline(
        metadata_connection_config=None, 
        pipeline_root=pipeline_root,
        num_epochs=data_types.RuntimeParameter(
            name='num_epochs',
            default=defaults.NUM_EPOCHS,
            ptype=int,
        ),
         batch_size=data_types.RuntimeParameter(
             name='batch_size',
             default=defaults.BATCH_SIZE,
             ptype=int,
        ),
         learning_rate=data_types.RuntimeParameter(
             name='learning_rate',
             default=defaults.LEARNING_RATE,
             ptype=float,
        ),
         hidden_units=data_types.RuntimeParameter(
             name='hidden_units',
             default=','.join(str(u) for u in defaults.HIDDEN_UNITS),
             ptype=str,
        ),
    )
    
    pipeline_definition_file = f'{config.PIPELINE_NAME}.json'

    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            project_id=config.PROJECT,
            default_image=config.IMAGE_URI
        ),
        output_filename=pipeline_definition_file)
    
    _ = runner.compile(managed_pipeline, write_out=True)
    return pipeline_definition_file
    
    
def submit_pipeline(pipeline_definition_file):
    
    pipeline_client = client.Client(
        project_id=config.PROJECT,
        region=config.REGION,
        api_key=config.API_KEY
    )

    pipeline_client.create_run_from_job_spec(
        pipeline_definition_file)



