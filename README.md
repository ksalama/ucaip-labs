# Code labs for Cloud AI Platform (Unified)

## Getting started

1. [Create a GCP Project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#console), [enable billing](https://cloud.google.com/billing/docs/how-to/modify-project), and [create a GCS bucket](https://cloud.google.com/storage/docs/creating-buckets).
2. [Enable the required APIs](https://cloud.google.com/endpoints/docs/openapi/enable-api).
3. Generate [API Key](https://cloud.google.com/docs/authentication/api-keys) to use for submitting AI Platfrom Managed Pipeline jobs.
4. [Create an AI Notebook instance](https://cloud.google.com/ai-platform/notebooks/docs/create-new).
5. Open the JupyterLab then open a new Terminal
6. Clone the repository to your AI Notebook instance:
```
git clone https://github.com/ksalama/ucaip-labs.git
cd ucaip-labs
```
7. Run the following commands to install the required packages:
```
pip install -r requirements.txt
```

## Data Analysis and Preparation

The [Chicago Taxi Trips](https://pantheon.corp.google.com/marketplace/details/city-of-chicago-public-data/chicago-taxi-trips) dataset is one ofof [public datasets hosted with BigQuery](https://cloud.google.com/bigquery/public-data/), which includes taxi trips from 2013 to the present, reported to the City of Chicago in its role as a regulatory agency. The task is to predict whether a given trip will result in a tip > 20%.

The [01-dataset-management](01-dataset-management.ipynb) notebook covers:
1. Performing exploratory data analysis on the data in BigQuery.
2. Creating managed AI Platform Dataset using the Python SDK.
3. Generating the schema for the raw data using [TensorFlow Data Validation](https://www.tensorflow.org/tfx/guide/tfdv).


## Experimentation

We experiment with creating a [Custom Model](https://cloud.google.com/ai-platform-unified/docs/training/create-model-custom-training) the using [02-experimentation](02-experimentation.ipynb) notebook, which covers:
1. Preparing the data using Dataflow
2. Implementing a Keras classification model
3. Training the Keras model in AI Platform using a [pre-built container](https://cloud.google.com/ai-platform-unified/docs/training/pre-built-containers)
4. Upload the exported model from Cloud Storage to AI Platform as a Model.

## Model Deployment
We use [Cloud Build](https://cloud.google.com/build) test and deploy the uploaded model to AI Plaform prediction.
The [03-model-deployment](02-model-deployment.ipynb) configures and executes the [build/model-deployment.yaml](build/model-deployment.yaml)
file with the following steps:
1. Creating an AI Platform Endpoint.
2. Test model interface.
3. Deploy the custom modesl to the endpoint.
4. Test the endpoint.

## Prediction Serving

We serve the deployed model for prediction. 
The [04-prediction-serving](04-prediction-serving.ipynb) notebook covers:

1. Use the endpoint for online prediction.
2. Use the uploaded model for batch prediciton.

## Training Operationalization

We build an end-to-end [TFX training pipeline](tfx_pipline) that performs the following steps:
1. Receive hyperparameters using hyperparam_gen custom python component.
2. Extract data from BigQuery using BigQueryExampleGen.
3. Validate the raw data using StatisticsGen and ExampleValidator.
4. Process the data using Transform.
5. Train a custom model using Trainer.
6. Evaluat and validate the custom model using ModelEvaluator.
7. Save the blessed to model registry location using using Pusher.
8. Upload the model to AI Platform using aip_model_pusher custom python component.

We have the following notebooks for the ML training pipeline:
1. The [05-tfx-interactive](05-tfx-interactive.ipynb) notebook covers testing the pipeline components interactively.
2. The [06-pipeline-deployment](06-pipeline-deployment.ipynb) notebook covers compiling and running the pipeline to AI Platform Managed Pipelines.

## Continuous Training

## (TODO) Model Monitoring

## (TODO) ML Metadata Tracking




