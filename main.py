import json
import os
from pdb import run
import time

import redis
import mlflow

from src.orchestrator.job_scheduler import JobScheduler
from utils.logging import logger
from src.kubernetes.k3s_builder import create_k3s_project, run_k3s_project


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

                    logger.info("Starting training pods in K3s cluster")

                    mlflow_experiment = job_data.get(
                        "mlflow_experiment", "automl-experiments"
                    )

                    project = create_k3s_project(
                        "automl-training",
                        job_scheduler.job_map,
                        env_vars={
                            "REQUEST_ID": request_id,
                            "MLFLOW_EXPERIMENT": mlflow_experiment,
                        },
                    )
                    run_k3s_project(project)
                    
                    logger.info("Waiting for training pods to complete (polling MLflow)...")
                    
                    while True:
                        try:
                            experiment = mlflow.get_experiment_by_name(mlflow_experiment)
                            if not experiment:
                                time.sleep(2)
                                continue
                                
                            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="", run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY)
                            
                            if runs.empty:
                                time.sleep(2)
                                continue
                            
                            active_runs = runs[runs['status'].isin(['RUNNING', 'SCHEDULED'])]
                            finished_runs = runs[runs['status'].isin(['FINISHED', 'FAILED', 'KILLED'])]
                            
                            total_visible_runs = len(active_runs) + len(finished_runs)
                            
                            if total_visible_runs > 0 and len(active_runs) == 0:
                                logger.info("All MLflow runs finished.")
                                break
                                
                            time.sleep(5)
                        except Exception as e:
                            logger.error(f"Error polling MLflow: {e}")
                            time.sleep(5)

                    if request_id:
                        redis_client.hset(
                            f"request:{request_id}",
                            mapping={
                                "status": "COMPLETED",
                                "message": "All training completed successfully",
                            },
                        )
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
