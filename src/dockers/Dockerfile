FROM python:3.11-slim

ARG DATASET_PATH
ARG MODEL_TYPE
ARG PARAMS
ARG IS_CLASSIFICATION
ARG TARGET

ENV DATASET_PATH=${DATASET_PATH}
ENV MODEL_TYPE=${MODEL_TYPE}
ENV PARAMS=${PARAMS}
ENV IS_CLASSIFICATION=${IS_CLASSIFICATION}
ENV TARGET=${TARGET}

WORKDIR /app

COPY ${DATASET_PATH} /app/data/
COPY training/training.py /app/training.py

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


CMD ["python", "training.py"]
