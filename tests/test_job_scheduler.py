"""
Unit tests for JobScheduler class that handles Docker image building and job mapping.
"""

import pytest
from unittest.mock import Mock, patch, call
from pathlib import Path
import subprocess

from src.orchestrator.job_scheduler import JobScheduler


@pytest.fixture
def job_scheduler():
    """Create a JobScheduler instance."""
    return JobScheduler()


@pytest.fixture
def sample_job_data():
    """Sample job data from Redis queue."""
    return {
        "dataset_path": "/path/to/data/iris.csv",
        "dataset_name": "iris_dataset",
        "target_name": "species",
        "task_type": "classification",
    }


class TestJobScheduler:
    """Test suite for JobScheduler class."""

    def test_initialization(self, job_scheduler):
        """Test JobScheduler initialization."""
        assert job_scheduler.job_map == {}
        assert job_scheduler.docker_container_tags == []

    def test_build_params_returns_all_models(self, job_scheduler):
        """Test that build_params returns parameters for all supported models."""
        params = job_scheduler.build_params()

        assert "linear_regression" in params
        assert "random_forest" in params
        assert "xgboost" in params
        assert "feed_forward_nn" in params

    def test_build_params_linear_regression(self, job_scheduler):
        """Test linear regression parameters."""
        params = job_scheduler.build_params()
        assert params["linear_regression"] == [{}]

    def test_build_params_random_forest(self, job_scheduler):
        """Test random forest parameters."""
        params = job_scheduler.build_params()
        assert isinstance(params["random_forest"], list)
        assert len(params["random_forest"]) == 3
        for p in params["random_forest"]:
            assert "n_estimators" in p
            assert "max_depth" in p

    def test_build_params_xgboost(self, job_scheduler):
        """Test XGBoost parameters."""
        params = job_scheduler.build_params()
        assert isinstance(params["xgboost"], list)
        assert len(params["xgboost"]) == 3
        for p in params["xgboost"]:
            assert "learning_rate" in p
            assert "n_estimators" in p
            assert p["is_xgboost"] is True

    def test_build_params_feed_forward_nn(self, job_scheduler):
        """Test feed forward neural network parameters."""
        params = job_scheduler.build_params()
        assert isinstance(params["feed_forward_nn"], list)
        assert len(params["feed_forward_nn"]) == 3
        for p in params["feed_forward_nn"]:
            assert "layers" in p
            assert p["output_dim"] == 2
            assert p["hidden_dim"] == 64
            assert p["input_dim"] == 32

    def test_get_model_instance_name_linear(self, job_scheduler):
        """Test getting model instance name for linear regression."""
        name = job_scheduler.get_model_instance_name("linear_regression")
        assert name == "Linear"

    def test_get_model_instance_name_random_forest(self, job_scheduler):
        """Test getting model instance name for random forest."""
        name = job_scheduler.get_model_instance_name("random_forest")
        assert name == "RandomForestClassifier"

    def test_get_model_instance_name_xgboost(self, job_scheduler):
        """Test getting model instance name for XGBoost."""
        name = job_scheduler.get_model_instance_name("xgboost")
        assert name == "Xgboost"

    def test_get_model_instance_name_nn(self, job_scheduler):
        """Test getting model instance name for neural network."""
        name = job_scheduler.get_model_instance_name("feed_forward_nn")
        assert name == "Sequential"

    def test_get_model_instance_name_unknown(self, job_scheduler):
        """Test getting model instance name for unknown model."""
        name = job_scheduler.get_model_instance_name("unknown_model")
        assert name == "UnknownModel"

    @patch("subprocess.run")
    def test_build_images_success(
        self, mock_subprocess, job_scheduler, sample_job_data
    ):
        """Test successful Docker image building."""
        # Arrange
        mock_subprocess.return_value = Mock(returncode=0)
        model_type = "random_forest"

        # Act
        job_scheduler.build_images(sample_job_data, model_type)

        # Assert
        assert len(job_scheduler.docker_container_tags) == 3
        assert job_scheduler.docker_container_tags[0].startswith(
            "automl_platform/random_forest_model_image_"
        )
        assert mock_subprocess.call_count == 3

        # Verify command structure for the first call
        call_args = mock_subprocess.call_args_list[0][0][0]
        assert call_args[0] == "docker"
        assert call_args[1] == "build"
        assert "-t" in call_args
        assert "automl_platform/random_forest_model_image_0" in call_args

    @patch("subprocess.run")
    def test_build_images_with_dataset_path(
        self, mock_subprocess, job_scheduler, sample_job_data
    ):
        """Test that dataset path is passed as build argument."""
        # Arrange
        mock_subprocess.return_value = Mock(returncode=0)

        # Act
        job_scheduler.build_images(sample_job_data, "linear_regression")

        # Assert
        call_args = mock_subprocess.call_args[0][0]
        # Check for DATA_PATH (the actual build arg name used)
        assert any(
            arg.startswith("DATA_PATH=") and arg.endswith("iris.csv")
            for arg in call_args
        )

    @patch("subprocess.run")
    def test_build_images_with_target_name(
        self, mock_subprocess, job_scheduler, sample_job_data
    ):
        """Test that target name is passed as build argument."""
        # Arrange
        mock_subprocess.return_value = Mock(returncode=0)

        # Act
        job_scheduler.build_images(sample_job_data, "xgboost")

        # Assert
        call_args = mock_subprocess.call_args[0][0]
        assert f"TARGET={sample_job_data['target_name']}" in call_args

    @patch("subprocess.run")
    def test_build_images_with_model_instance_name(
        self, mock_subprocess, job_scheduler, sample_job_data
    ):
        """Test that correct model instance name is passed."""
        # Arrange
        mock_subprocess.return_value = Mock(returncode=0)

        # Act
        job_scheduler.build_images(sample_job_data, "random_forest")

        # Assert
        call_args = mock_subprocess.call_args[0][0]
        assert "MODEL_TYPE=RandomForestClassifier" in call_args

    @patch("subprocess.run")
    def test_build_images_subprocess_error(
        self, mock_subprocess, job_scheduler, sample_job_data
    ):
        """Test handling of subprocess error during image building."""
        # Arrange
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "docker build")

        # Act & Assert
        with pytest.raises(subprocess.CalledProcessError):
            job_scheduler.build_images(sample_job_data, "linear_regression")

    def test_fill_job_map_success(self, job_scheduler):
        """Test successful job map filling."""
        # Arrange
        job_scheduler.docker_container_tags = ["automl_platform/test_image_0"]

        # Act
        job_scheduler.fill_job_map()

        # Assert
        assert len(job_scheduler.job_map) == 1
        job_id = list(job_scheduler.job_map.keys())[0]
        assert job_scheduler.job_map[job_id] == "automl_platform/test_image_0"

    def test_fill_job_map_without_docker_tag(self, job_scheduler):
        """Test fill_job_map when docker_container_tags is empty."""
        # Arrange
        job_scheduler.docker_container_tags = []

        # Act
        job_scheduler.fill_job_map()

        # Assert
        assert len(job_scheduler.job_map) == 0

    def test_fill_job_map_multiple_times(self, job_scheduler):
        """Test filling job map multiple times creates unique job IDs."""
        # Arrange
        job_scheduler.docker_container_tags = ["automl_platform/model1"]

        # Act
        job_scheduler.fill_job_map()
        first_job_id = list(job_scheduler.job_map.keys())[0]

        job_scheduler.docker_container_tags = ["automl_platform/model2"]
        job_scheduler.fill_job_map()

        # Assert
        assert len(job_scheduler.job_map) == 2
        job_ids = list(job_scheduler.job_map.keys())
        assert job_ids[0] != job_ids[1]  # Unique UUIDs

    def test_get_job_map(self, job_scheduler):
        """Test retrieving the job map."""
        # Arrange
        job_scheduler.job_map = {"job-123": "image:tag"}

        # Act
        result = job_scheduler.get_job_map()

        # Assert
        assert result == {"job-123": "image:tag"}

    def test_get_job_map_empty(self, job_scheduler):
        """Test retrieving empty job map."""
        # Act
        result = job_scheduler.get_job_map()

        # Assert
        assert result == {}


class TestJobSchedulerIntegration:
    """Integration tests for complete JobScheduler workflow."""

    @patch("subprocess.run")
    def test_full_workflow_single_model(
        self, mock_subprocess, job_scheduler, sample_job_data
    ):
        """Test complete workflow: build image -> fill job map."""
        # Arrange
        mock_subprocess.return_value = Mock(returncode=0)
        model_type = "xgboost"

        # Act
        job_scheduler.build_images(sample_job_data, model_type)
        job_scheduler.fill_job_map()

        # Assert
        # xgboost now builds 3 images
        assert len(job_scheduler.job_map) == 3
        assert len(job_scheduler.docker_container_tags) == 3
        assert job_scheduler.docker_container_tags[0].startswith(
            "automl_platform/xgboost_model_image_"
        )

        job_id = list(job_scheduler.job_map.keys())[0]
        assert job_scheduler.job_map[job_id].startswith(
            "automl_platform/xgboost_model_image_"
        )

    @patch("subprocess.run")
    def test_full_workflow_multiple_models(
        self, mock_subprocess, job_scheduler, sample_job_data
    ):
        """Test workflow with multiple models."""
        # Arrange
        mock_subprocess.return_value = Mock(returncode=0)
        models = ["linear_regression", "random_forest", "xgboost"]

        # Act
        for model in models:
            job_scheduler.build_images(sample_job_data, model)
            job_scheduler.fill_job_map()

        # Assert
        # linear_regression: 1
        # random_forest: 3
        # xgboost: 3
        # Total: 7
        assert len(job_scheduler.job_map) == 7

        # Verify all unique job IDs
        job_ids = list(job_scheduler.job_map.keys())
        assert len(job_ids) == len(set(job_ids))
