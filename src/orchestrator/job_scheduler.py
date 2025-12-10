from uuid import uuid4
import subprocess
import random
import os
import json

from utils.logging import logger


class JobScheduler:
    """
    Singleton Job Manager class to handle all training jobs
    """

    def __init__(self):
        self.job_map = {}
        self.docker_container_tags = []

    def build_params(self) -> dict:
        """
        Build the parameters for the different models.
        Currently chooses 3 random sets of parameters for each model.

        Returns:
            dict: Parameters for the models
        """
        logger.info("Building model parameters")
        num_configs = 2
        n_estimators_range = list(range(50, 201, 10))
        max_depth_range = list(range(5, 21, 1))
        learning_rate_range = [0.01, 0.05, 0.1, 0.2, 0.3]
        hidden_layers_range = list(range(1, 21, 1))

        rf_params = [
            {
                "n_estimators": random.choice(n_estimators_range),
                "max_depth": random.choice(max_depth_range),
            }
            for _ in range(num_configs)
        ]

        xgb_params = [
            {
                "learning_rate": random.choice(learning_rate_range),
                "n_estimators": random.choice(n_estimators_range),
                "is_xgboost": True,
            }
            for _ in range(num_configs)
        ]

        nn_params = [
            {
                "layers": random.choice(hidden_layers_range),
                "output_dim": 2,
                "hidden_dim": 64,
                "input_dim": 32,
            }
            for _ in range(num_configs)
        ]

        params = {
            "linear_regression": [
                {"in_features": 32, "out_features": 2} for _ in range(num_configs)
            ],
            "random_forest": rf_params,
            "xgboost": xgb_params,
            "feed_forward_nn": nn_params,
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

    def build_images(
        self, job_data: dict, model_type: str, progress_callback=None
    ) -> None:
        """
        Build the docker images for the different models.
        """
        logger.info("Building Docker images for models")
        dataset_path = job_data.get("dataset_path", "")

        if os.path.isabs(dataset_path):
            try:
                dataset_path = os.path.relpath(dataset_path)
            except ValueError:
                logger.warning(
                    "Could not convert dataset path to relative path: %s", dataset_path
                )

        dataset_filename = os.path.basename(dataset_path)

        target = job_data.get("target_name", "target")

        model_params_list = self.build_params().get(model_type, [{}])
        total_images = len(model_params_list)

        model_instance_name = self.get_model_instance_name(model_type)
        is_classification = job_data.get("is_classification", 0)

        self.docker_container_tags = []

        for i, params in enumerate(model_params_list):
            image_tag = f"automl_platform/{model_type.lower()}_model_image_{i}"

            logger.info(
                "Building image %d for model type: %s with params: %s",
                i,
                model_type,
                params,
            )

            image_build_cmd = [
                "docker",
                "build",
                "-f",
                "src/dockers/Dockerfile",
                "--build-arg",
                f"DATA_PATH={dataset_path}",
                "--build-arg",
                f"DATASET_FILENAME={dataset_filename}",
                "--build-arg",
                f"MODEL_TYPE={model_instance_name}",
                "--build-arg",
                f"TARGET={target}",
                "--build-arg",
                f"PARAMS={json.dumps(params)}",
                "--build-arg",
                f"MLFLOW_EXPERIMENT={job_data.get('mlflow_experiment', 'Default')}",
                "--build-arg",
                f"IS_CLASSIFICATION={is_classification}",
                "-t",
                image_tag,
                ".",
            ]

            subprocess.run(image_build_cmd, check=True)
            logger.info("Docker image built with tag: %s", image_tag)
            self.docker_container_tags.append(image_tag)

            if progress_callback:
                progress_callback(model_type, i + 1, total_images)

    def fill_job_map(self) -> None:
        """
        Fill the hash map that contains key : value pairs of job_id : docker_container_tag
        this map will be used by the kubernetes compo
        """
        logger.info("Filling job map with job IDs and Docker container tags")
        if self.docker_container_tags:
            for _, tag in enumerate(self.docker_container_tags):
                model_part = tag.split("/")[-1].replace("_", "-")
                job_id = f"job-{model_part}-{str(uuid4())[:8]}"

                self.job_map[job_id] = tag
            logger.info("Job map updated: %s", self.job_map)
        else:
            logger.error(
                "No Docker container tags set. Cannot fill job map. Make sure the images were built"
            )

    def get_job_map(self) -> dict:
        """
        Get the job map containing job IDs and Docker container tags.

        Returns:
            dict: The job map
        """
        return self.job_map

    def run_containers(self) -> None:
        """
        Run all Docker containers that have been built.
        Mounts the mlruns directory so containers write directly to filesystem.
        """
        logger.info("Running Docker containers for training")

        mlruns_path = os.path.abspath("./mlruns")

        for job_id, image_tag in self.job_map.items():
            logger.info("Running container for job %s with image %s", job_id, image_tag)

            run_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{mlruns_path}:{mlruns_path}",
                "-e",
                f"MLFLOW_TRACKING_URI=file://{mlruns_path}",
            ]

            run_cmd.append(image_tag)

            try:
                result = subprocess.run(
                    run_cmd, check=True, capture_output=True, text=True
                )
                logger.info("Container %s completed successfully", job_id)
                logger.debug("Container output: %s", result.stdout)
            except subprocess.CalledProcessError as e:
                logger.error("Container %s failed with error: %s", job_id, e.stderr)
                continue
