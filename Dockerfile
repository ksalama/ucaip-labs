FROM gcr.io/tfx-oss-public/tfx:0.27.0

RUN pip install google-cloud-aiplatform google-cloud-automl

COPY model_src/ model_src/
COPY tfx_pipeline/ tfx_pipeline/
COPY utils/ utils/
COPY __init__.py __init__.py

ENV PYTHONPATH="/pipeline:${PYTHONPATH}"