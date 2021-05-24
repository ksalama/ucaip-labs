FROM gcr.io/tfx-oss-public/tfx:0.30.0

RUN pip install google-cloud-aiplatform apache-beam==2.29.0

COPY src/ src/

ENV PYTHONPATH="/pipeline:${PYTHONPATH}"