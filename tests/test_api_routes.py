"""
Unit tests for API routes that handle dataset submission and Redis queue integration.
"""

import json
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
from fastapi.testclient import TestClient

from src.api.dto.datasetRequest import DatasetRequest
from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_dataset_request():
    """Sample dataset request payload."""
    return {
        "name": "iris_dataset",
        "target_name": "species",
        "task_type": "classification",
        "dataset_csv": "sepal_length,sepal_width,species\n5.1,3.5,setosa\n4.9,3.0,setosa\n",
    }


@pytest.fixture
def sample_dataset_request_regression():
    """Sample dataset request for regression task."""
    return {
        "name": "housing_prices",
        "target_name": "price",
        "task_type": "regression",
        "dataset_csv": "rooms,area,price\n3,1200,250000\n4,1500,320000\n",
    }


class TestDatasetEndpoint:
    """Test suite for /d_dataset endpoint."""

    @patch("src.api.routes.core.redis_client")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_send_dataset_success(
        self, mock_mkdir, mock_file, mock_redis, client, sample_dataset_request
    ):
        """Test successful dataset submission and Redis queue publishing."""
        # Arrange
        mock_redis.lpush = Mock(return_value=1)

        # Act
        response = client.post("/api/v1/d_dataset", json=sample_dataset_request)

        # Assert
        assert response.status_code == 200
        assert (
            response.json()["message"]
            == "Dataset received successfully, job queued for training"
        )

        # Verify file was written
        mock_file.assert_called_once()
        written_data = "".join(
            call.args[0] for call in mock_file().write.call_args_list
        )
        assert written_data == sample_dataset_request["dataset_csv"]

        # Verify Redis queue was called
        mock_redis.lpush.assert_called_once()
        call_args = mock_redis.lpush.call_args
        assert call_args[0][0] == "job_queue"

        # Verify job data structure
        job_data = json.loads(call_args[0][1])
        assert job_data["dataset_name"] == "iris_dataset"
        assert job_data["target_name"] == "species"
        assert job_data["task_type"] == "classification"
        assert "dataset_path" in job_data

    @patch("src.api.routes.core.redis_client")
    def test_send_dataset_missing_csv(self, mock_redis, client):
        """Test endpoint with missing dataset CSV."""
        # Arrange
        invalid_request = {
            "name": "test_dataset",
            "target_name": "target",
            "task_type": "classification",
            "dataset_csv": None,
        }

        # Act
        response = client.post("/api/v1/d_dataset", json=invalid_request)

        # Assert
        assert response.status_code == 400
        assert "Bad Request" in response.json()["detail"]

    @patch("src.api.routes.core.redis_client", None)
    def test_send_dataset_redis_unavailable(self, client, sample_dataset_request):
        """Test endpoint when Redis is unavailable."""
        # Act
        response = client.post("/api/v1/d_dataset", json=sample_dataset_request)

        # Assert
        assert response.status_code == 503
        assert "Redis queue is unavailable" in response.json()["detail"]

    @patch("src.api.routes.core.redis_client")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_send_dataset_redis_publish_fails(
        self, mock_mkdir, mock_file, mock_redis, client, sample_dataset_request
    ):
        """Test handling of Redis publish failure."""
        # Arrange
        mock_redis.lpush = Mock(side_effect=Exception("Redis connection lost"))

        # Act
        response = client.post("/api/v1/d_dataset", json=sample_dataset_request)

        # Assert
        assert response.status_code == 500
        assert "Failed to queue job" in response.json()["detail"]

    @patch("src.api.routes.core.redis_client")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_send_dataset_regression_task(
        self,
        mock_mkdir,
        mock_file,
        mock_redis,
        client,
        sample_dataset_request_regression,
    ):
        """Test dataset submission for regression task."""
        # Arrange
        mock_redis.lpush = Mock(return_value=1)

        # Act
        response = client.post(
            "/api/v1/d_dataset", json=sample_dataset_request_regression
        )

        # Assert
        assert response.status_code == 200

        # Verify job data has correct task type
        call_args = mock_redis.lpush.call_args
        job_data = json.loads(call_args[0][1])
        assert job_data["task_type"] == "regression"

    @patch("src.api.routes.core.redis_client")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_send_dataset_creates_storage_directory(
        self, mock_mkdir, mock_file, mock_redis, client, sample_dataset_request
    ):
        """Test that data_storage directory is created."""
        # Arrange
        mock_redis.lpush = Mock(return_value=1)

        # Act
        response = client.post("/api/v1/d_dataset", json=sample_dataset_request)

        # Assert
        assert response.status_code == 200
        mock_mkdir.assert_called_once_with(exist_ok=True)


class TestDatasetRequestModel:
    """Test suite for DatasetRequest DTO validation."""

    def test_valid_dataset_request_classification(self):
        """Test valid classification dataset request."""
        data = {
            "name": "test_data",
            "target_name": "label",
            "task_type": "classification",
            "dataset_csv": "col1,col2\n1,2\n",
        }
        request = DatasetRequest(**data)
        assert request.name == "test_data"
        assert request.task_type == "classification"

    def test_valid_dataset_request_regression(self):
        """Test valid regression dataset request."""
        data = {
            "name": "test_data",
            "target_name": "value",
            "task_type": "regression",
            "dataset_csv": "col1,col2\n1,2\n",
        }
        request = DatasetRequest(**data)
        assert request.task_type == "regression"

    def test_default_task_type(self):
        """Test default task type is classification."""
        data = {
            "name": "test_data",
            "target_name": "label",
            "dataset_csv": "col1,col2\n1,2\n",
        }
        request = DatasetRequest(**data)
        assert request.task_type == "classification"

    def test_invalid_task_type(self):
        """Test invalid task type raises validation error."""
        data = {
            "name": "test_data",
            "target_name": "label",
            "task_type": "clustering",  # Invalid
            "dataset_csv": "col1,col2\n1,2\n",
        }
        with pytest.raises(Exception):  # Pydantic validation error
            DatasetRequest(**data)

    def test_missing_required_fields(self):
        """Test missing required fields raises validation error."""
        data = {
            "name": "test_data",
            # Missing target_name
        }
        with pytest.raises(Exception):
            DatasetRequest(**data)
