FROM gcr.io/tfx-oss-public/tfx:0.30.0

RUN pip install google-cloud-aiplatform

COPY src/ src/

ENV PYTHONPATH="/pipeline:${PYTHONPATH}"