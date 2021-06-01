# (WIP) Setting up MLOps environment on Google Cloud

1. [Create a GCP Project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#console), [enable billing](https://cloud.google.com/billing/docs/how-to/modify-project), and [create a GCS bucket](https://cloud.google.com/storage/docs/creating-buckets).
2. [Enable the required APIs](https://cloud.google.com/endpoints/docs/openapi/enable-api).
3. [Create an AI Notebook instance](https://cloud.google.com/ai-platform/notebooks/docs/create-new).

TODOs: Automate the environmentsetup using Terraform

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
