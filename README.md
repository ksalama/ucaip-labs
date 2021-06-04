# (WIP) MLOps on Vertex AI

This example implements the end-to-end [MLOps process](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf) using [Vertex AI](https://cloud.google.com/vertex-ai) platform and [Smart Analytics](https://cloud.google.com/solutions/smart-analytics) technology capabilities. The example use [Keras](https://keras.io/) to implement the ML model, [TFX](https://www.tensorflow.org/tfx) to implement the training pipeline, and [Model Builder SDK](https://github.com/googleapis/python-aiplatform/tree/569d4cd03e888fde0171f7b0060695a14f99b072/google/cloud/aiplatform) to interact with Vertex AI.


<img src="mlops.png" alt="MLOps lifecycle" width="400"/>


## Getting started

1. [Setting up MLOps environment](provision) on Google Cloud.
2. Start your AI Notebook instance.
3. Open the JupyterLab then open a new Terminal
4. Clone the repository to your AI Notebook instance:
    ```
    git clone https://github.com/ksalama/ucaip-labs.git
    cd ucaip-labs
    ```
5. Run the following commands to install the required packages:
    ```
    pip install tfx==0.30.0
    pip install tensorflow==2.4.1
    pip install -r requirements.txt
    ```

## Dataset Management

The [Chicago Taxi Trips](https://pantheon.corp.google.com/marketplace/details/city-of-chicago-public-data/chicago-taxi-trips) dataset is one ofof [public datasets hosted with BigQuery](https://cloud.google.com/bigquery/public-data/), which includes taxi trips from 2013 to the present, reported to the City of Chicago in its role as a regulatory agency. The task is to predict whether a given trip will result in a tip > 20%.

The [01-dataset-management](01-dataset-management.ipynb) notebook covers:
1. Performing exploratory data analysis on the data in BigQuery.
2. Creating managed Vertex AI Dataset using the Python SDK.
3. Generating the schema for the raw data using [TensorFlow Data Validation](https://www.tensorflow.org/tfx/guide/tfdv).


## ML Development

We experiment with creating a [Custom Model](https://cloud.google.com/ai-platform-unified/docs/training/create-model-custom-training) using [02-experimentation](02-experimentation.ipynb) notebook, which covers:
1. Preparing the data using Dataflow.
2. Implementing a Keras classification model.
3. Training the Keras model in Vertex AI using a [pre-built container](https://cloud.google.com/ai-platform-unified/docs/training/pre-built-containers).
4. Upload the exported model from Cloud Storage to Vertex AI as a Model.
5. Exract and visualize experiment parameters from [Vertex AI Metadata](https://cloud.google.com/vertex-ai/docs/ml-metadata/introduction).

We use [Vertex TensorBoard](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview) 
and [Vertex ML Metadata](https://cloud.google.com/vertex-ai/docs/ml-metadata/introduction) to  track, visualize, and compare ML experiments.

In addition, the training steps are formalized by implementing a [TFX pipeline](https://www.tensorflow.org/tfx).
The [03-training-formalization](02-tfx-interactive.ipynb) notebook covers implementing and testing the pipeline components interactively.

## Training Operationalization

The end-to-end TFX training pipeline implementation is in the [src/pipelines](src/tfx_pipelines) directory, which covers the following steps:
1. Receive hyperparameters using hyperparam_gen custom python component.
2. Extract data from BigQuery using BigQueryExampleGen.
3. Validate the raw data using StatisticsGen and ExampleValidator.
4. Process the data using Transform.
5. Train a custom model using Trainer.
6. Evaluat and validate the custom model using ModelEvaluator.
7. Save the blessed to model registry location using using Pusher.
8. Upload the model to Vertex AI using aip_model_pusher custom python component.

The [04-pipeline-deployment](04-pipeline-deployment.ipynb) notebook covers testing, compiling, and running the pipeline locally and using [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction?hl=nn).

## Continuous Training

After testing, compiling, and uploading the pipeline definition to Cloud Storage, the pipeline is executed with respect to a trigger. 
We use [Cloud Functions](https://cloud.google.com/functions) and [Cloud Pub/Sub](https://cloud.google.com/pubsub) as a triggering mechanism. 

The [05-continuous-training](05-continuous-training.ipynb) notebook covers the following steps:
1. Create the Cloud Pub/Sub topic.
2. Deploy the Cloud Function, which is implemented in [src/pipeline_triggering](src/pipeline_triggering).
3. Test triggering a pipeline.


## Model Deployment

We use [Cloud Build](https://cloud.google.com/build) test and deploy the uploaded model to [Vertex AI Prediction](https://cloud.google.com/vertex-ai/docs/predictions/getting-predictions?hl=nn).
The [06-model-deployment](06-model-deployment.ipynb) configures and executes the [build/model-deployment.yaml](build/model-deployment.yaml)
file with the following steps:
1. Creating an Vertex AI Endpoint.
2. Test model interface.
3. Create an endpoint in Vertex AI.
4. Deploy the model to the endpoint.
5. Test the endpoint.

## Prediction Serving

We serve the deployed model for prediction. 
The [07-prediction-serving](07-prediction-serving.ipynb) notebook covers:

1. Use the endpoint for online prediction.
2. Use the uploaded model for batch prediciton.

## Model Monitoring

After a model is deployed in for prediciton serving, continuous monitoring is set up to ensure that the model continue to perform as expected.
The [08-model-monitoring](08-model-monitoring.ipynb) notebook covers configuring [Vertex AI Model Monitoring](https://cloud.google.com/vertex-ai/docs/model-monitoring/overview?hl=nn) for skew and dirft detection:
1. Set skew and drift threshold.
2. Create a monitoring job for all the models under and endpoint.
3. List the monitoring jobs.
4. List artifacts produced by monitoring job.
5. Pause and delete the monitoring job.


## Metadata Tracking

You can view the parameters and metrics logged by your experiments, as well as the artifacts and metadata stored by 
your Vertex AI Pipelines in [Cloud Console](https://console.cloud.google.com/vertex-ai/metadata).

## Disclaimer

This is not an official Google product but sample code provided for an educational purpose.

---

Copyright 2021 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.






