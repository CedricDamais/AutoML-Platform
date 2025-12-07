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
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=0,
            decode_responses=True,
        )
        redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e)
        return

    # Configuration
    enable_gpu = os.getenv("ENABLE_GPU", "false").lower() == "true"

    logger.info("Initializing Job Scheduler (GPU: %s)", enable_gpu)
    job_scheduler = JobScheduler(enable_gpu=enable_gpu)

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
                request_id = job_data.get("request_id")

                if request_id:
                    redis_client.hset(
                        f"request:{request_id}",
                        mapping={
                            "status": "PROCESSING",
                            "message": "Job picked up by worker",
                        },
                    )

                def update_progress(model_type, current, total):
                    if request_id:
                        redis_client.hset(
                            f"request:{request_id}",
                            mapping={
                                f"progress_{model_type}": f"{current}/{total}",
                                "current_model": model_type,
                            },
                        )

                try:
                    is_classification = (
                        1 if job_data.get("task_type") == "classification" else 0
                    )
                    job_data["is_classification"] = is_classification

                    for model_type in models_to_build:
                        if request_id:
                            redis_client.hset(
                                f"request:{request_id}",
                                mapping={
                                    "status": "BUILDING",
                                    "message": f"Building images for {model_type}",
                                },
                            )

                        logger.info("Received job from queue: %s", job_data)
                        job_scheduler.build_images(
                            job_data, model_type, progress_callback=update_progress
                        )
                        logger.info("Built Docker image for model: %s", model_type)
                        job_scheduler.fill_job_map()
                        logger.info("Current job map: %s", job_scheduler.job_map)

                    if request_id:
                        redis_client.hset(
                            f"request:{request_id}",
                            mapping={
                                "status": "TRAINING",
                                "message": "Running training containers",
                            },
                        )

                    logger.info("Starting training containers...")
                    job_scheduler.run_containers()
                    logger.info("All training containers completed")

                    if request_id:
                        redis_client.hset(
                            f"request:{request_id}",
                            mapping={
                                "status": "COMPLETED",
                                "message": "All training completed successfully",
                            },
                        )
                        # Store the final job map in Redis for the dashboard to see
                        redis_client.hset(
                            f"request:{request_id}",
                            mapping={"job_map": json.dumps(job_scheduler.job_map)},
                        )

                except Exception as e:
                    logger.error("Error processing job: %s", e)
                    if request_id:
                        redis_client.hset(
                            f"request:{request_id}",
                            mapping={"status": "FAILED", "message": str(e)},
                        )
                    continue

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
