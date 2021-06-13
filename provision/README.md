# (WIP) Setting up MLOps environment on Google Cloud

1. [Create a GCP Project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#console) and [enable billing](https://cloud.google.com/billing/docs/how-to/modify-project).
2. [Create a Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets).
3. [Enable the required APIs](https://cloud.google.com/endpoints/docs/openapi/enable-api).
4. [Create an AI Notebook instance](https://cloud.google.com/ai-platform/notebooks/docs/create-new).
5. [Create a Vertex TensorBoard instance](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-training).
6. [Create a Service Account](https://cloud.google.com/iam/docs/creating-managing-service-accounts) with permissions to Vertex AI.
7. Set permissions to [Cloud Build service account](https://cloud.google.com/build/docs/securing-builds/configure-access-for-cloud-build-service-account) as Vertex AI User and Big Query User.


TODOs: Automate the environment setup using Terraform

1. Enable APIs: Vertex AI, BigQuery, Dataflow, PubSub, Cloud Function, Container Registry, Cloud Build
2. Create a Cloud Storage bucket.
3. Create managed TensorBoard instance.
4. Grant Access for service accounts
    *  AI Platform Custom Code Service Agent > Vertex AI Viewer
    *  Cloud Build > Vertex AI User, BigQuery User
    *  AI Platform Service Agent > Dataflow User, BigQuery User

5. Build AI Notebook Instance base container image
6. Create AI Notebook Instance
7. Build Cloud Build container image 
