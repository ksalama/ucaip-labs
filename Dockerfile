FROM gcr.io/tfx-oss-public/tfx:0.27.0

RUN pip install google-cloud-aiplatform google-cloud-automl

COPY src/ src/

ENV PYTHONPATH="/pipeline:${PYTHONPATH}"