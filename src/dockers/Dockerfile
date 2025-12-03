FROM python:3.11-slim

DATASET_PATH=./data/
MODEL_TYPE=""
PARAMS={}
IS_CLASSIFICATION=0
TARGET=""


WORKDIR /app
COPY ${LOCAL_DATA_PATH} /app/data/
COPY training/training.py /app/training.py

RUN pip freeze > requirements.txt
RUN pip install -r requirements.txt

