import json
import os
import time

import redis

from src.orchestrator.job_scheduler import JobScheduler
from utils.logging import logger


def main():
    """Main worker process that polls Redis queue and processes jobs"""
    logger.info("Starting AutoML Platform Job Worker")

    try:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True,
        )
        redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e)
        return

    logger.info("Initializing Job Scheduler")
    job_scheduler = JobScheduler()

    models_to_build = [
        "linear_regression",
        "random_forest",
        "xgboost",
        "feed_forward_nn",
    ]

    while True:
        try:
            result = redis_client.brpop("job_queue", timeout=1)

            if result is not None:
                _, job_data_json = result
                job_data = json.loads(job_data_json)

                for model_type in models_to_build:
                    logger.info("Received job from queue: %s", job_data)
                    job_scheduler.build_images(job_data, model_type)
                    logger.info("Built Docker image for model: %s", model_type)
                    job_scheduler.fill_job_map()
                    logger.info("Current job map: %s", job_scheduler.job_map)

            else:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal. Exiting...")
            break
        except Exception as e:
            logger.error("Error processing job: %s", e)
            time.sleep(1)


if __name__ == "__main__":
    main()
