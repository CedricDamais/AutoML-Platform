FROM python:3.9-slim

WORKDIR /app

# Install system dependencies, without these many issues arose,
# especially with packages that need compilation like xgboost
# git was also needed for mlflow to pull some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY src/dockers/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data_storage/ data_storage/

ENV PYTHONPATH=/app/src

ARG DATA_PATH
ARG DATASET_FILENAME
ARG MODEL_TYPE
ARG TARGET
ARG PARAMS
ARG IS_CLASSIFICATION

ENV DATA_PATH=${DATA_PATH}
ENV DATASET_FILENAME=${DATASET_FILENAME}
ENV MODEL=${MODEL_TYPE}
ENV TARGET=${TARGET}
ENV PARAMS=${PARAMS}
ENV IS_CLASSIFICATION=${IS_CLASSIFICATION}

ENTRYPOINT [ "python", "src/training/training.py" ]
