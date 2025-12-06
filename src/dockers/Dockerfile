FROM python:3.11-slim

WORKDIR /app

COPY src/dockers/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/training/training.py /app/training.py

ARG DATASET_PATH
ARG MODEL_TYPE
ARG PARAMS
ARG IS_CLASSIFICATION
ARG TARGET

COPY ${DATASET_PATH} /app/data/

ENV DATASET_PATH=${DATASET_PATH}
ENV MODEL_TYPE=${MODEL_TYPE}
ENV PARAMS=${PARAMS}
ENV IS_CLASSIFICATION=${IS_CLASSIFICATION}
ENV TARGET=${TARGET}

CMD ["python", "training.py"]
