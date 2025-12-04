from uuid import uuid4
import os
import subprocess
from pathlib import Path

from utils.logging import logger


class JobScheduler:
    """
    Singleton Job Manager class to handle all training jobs
    """

    def __init__(self):
        self.job_map = {}
        self.docker_container_tag = None

    def build_params(self) -> dict:
        """
        Build the parameters for the different models.

        Returns:
            dict: Parameters for the models
        """
        logger.info("Building model parameters")
        params = {
            "linear_regression": {},
            "random_forest": {"n_estimators": 100, "max_depth": 10},
            "xgboost": {"learning_rate": 0.1, "n_estimators": 100, "is_xgboost": True},
            "feed_forward_nn": {
                "layers": 10,
                "output_dim": 2,
                "hidden_dim": 64,
                "input_dim": 32,
            },
        }
        return params

    def get_model_instance_name(self, model_type: str) -> str:
        """
        Get the model instance name based on the model type.

        Args:
            model_type (str): The type of the model

        Returns:
            str: The model instance name
        """
        model_names = {
            "linear_regression": "Linear",
            "random_forest": "RandomForestClassifier",
            "xgboost": "Xgboost",
            "feed_forward_nn": "Sequential",
        }
        return model_names.get(model_type, "UnknownModel")

    def build_images(self, job_data: dict, model_type: str) -> None:
        """
        Build the docker images for the different models.
        """
        logger.info("Building Docker images for models")
        dataset_path = job_data.get("dataset_path", "")
        target = job_data.get("target_name", "target")
        params = self.build_params().get(model_type, {})
        is_classification = 0
        image_tag = f"automl_platform/{model_type.lower()}_model_image"
        model_instance_name = self.get_model_instance_name(model_type)

        logger.info("All parameters built for image creation")
        logger.info("Building image for model type: %s", model_type)

        image_build_cmd = [
            "docker",
            "build",
            "--build-arg",
            f"DATASET_PATH={dataset_path}",
            "--build-arg",
            f"MODEL_TYPE={model_instance_name}",
            "--build-arg",
            f"TARGET={target}",
            "--build-arg",
            f"PARAMS={params}",
            "--build-arg",
            f"IS_CLASSIFICATION={is_classification}",
            "-t",
            image_tag,
            "dockers/",
        ]
        self.docker_container_tag = image_tag
        subprocess.run(image_build_cmd, check=True)
        logger.info("Docker image built with tag: %s", image_tag)

    def fill_job_map(self) -> None:
        """
        Fill the hash map that contains key : value pairs of job_id : docker_container_tag
        this map will be used by the kubernetes compo
        """
        logger.info("Filling job map with job IDs and Docker container tags")
        if self.docker_container_tag:
            job_id = str(uuid4())

            self.job_map[job_id] = self.docker_container_tag
            logger.info("Job map filled: %s", self.job_map)
        else:
            logger.error(
                "Docker container tag is not set. Cannot fill job map. Make sure the images were built"
            )

    def get_job_map(self) -> dict:
        """
        Get the job map containing job IDs and Docker container tags.

        Returns:
            dict: The job map
        """
        return self.job_map
