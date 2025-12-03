import json
import os
from pathlib import Path

import redis
from fastapi import APIRouter, HTTPException

from utils.logging import logger

from ..dto.datasetRequest import DatasetRequest

router = APIRouter()

try:
    redis_client: redis.Redis = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=0,
        decode_responses=True,
    )
    redis_client.ping()
    logger.info("Connected to Redis successfully")
except redis.RedisError as e:
    logger.error("Failed to connect to Redis: %s", e)
    redis_client = None


@router.post("/d_dataset")
async def send_dataset(req: DatasetRequest):
    """
    The user of the app sends the dataset through this endpoint.
    We should keep the dataset size under 10MB to not have any timeout issues
    """
    if not req.dataset_csv:
        raise HTTPException(
            status_code=400,
            detail="Bad Request, You did not give a dataset to train on !",
        )

    if not redis_client:
        raise HTTPException(
            status_code=503,
            detail="Redis queue is unavailable. Cannot process any job.",
        )

    logger.info("Storing dataset sent by user")
    data_storage_path = Path("data_storage")
    data_storage_path.mkdir(exist_ok=True)

    logger.debug("Preparing to store dataset")

    path_to_store_dataset = data_storage_path / f"{req.name}.csv"

    logger.debug("Received Data set from user")
    logger.info(
        "Storing dataset to local storage at this path: %s", path_to_store_dataset
    )
    with open(path_to_store_dataset, "w", encoding="utf-8") as f:
        f.write(req.dataset_csv)

    job_data = {
        "dataset_path": str(path_to_store_dataset.absolute()),
        "dataset_name": req.name,
        "target_name": req.target_name,
        "task_type": req.task_type,
    }

    try:
        redis_client.lpush("job_queue", json.dumps(job_data))
        logger.info("Job published to Redis queue: %s", job_data)
    except Exception as e:
        logger.error("Failed to publish job to Redis: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to queue job for processing",
        ) from e

    logger.debug("Job successfully queued for processing")
    return {
        "message": "Dataset received successfully, job queued for training",
    }
