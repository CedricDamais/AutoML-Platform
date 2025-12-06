"""
Integration tests for the complete Redis queue workflow.
Tests the interaction between API, Redis queue, and Job Worker.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
import fakeredis

from src.orchestrator.job_scheduler import JobScheduler


@pytest.fixture
def fake_redis_server():
    """Create a fake Redis server for testing."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def sample_job_payload():
    """Sample job payload that would be sent through Redis."""
    return {
        "dataset_path": "/tmp/test_dataset.csv",
        "dataset_name": "test_dataset",
        "target_name": "target_column",
        "task_type": "classification",
    }


class TestRedisQueueIntegration:
    """Integration tests for Redis queue workflow."""

    def test_job_enqueue_dequeue(self, fake_redis_server, sample_job_payload):
        """Test that jobs can be enqueued and dequeued from Redis."""
        # Enqueue job
        fake_redis_server.lpush("job_queue", json.dumps(sample_job_payload))

        # Dequeue job
        _, job_data_json = fake_redis_server.brpop("job_queue", timeout=1)
        job_data = json.loads(job_data_json)

        # Assert
        assert job_data == sample_job_payload
        assert job_data["dataset_name"] == "test_dataset"
        assert job_data["task_type"] == "classification"

    def test_multiple_jobs_fifo_order(self, fake_redis_server):
        """Test that multiple jobs are processed in FIFO order."""
        # Enqueue multiple jobs
        jobs = [
            {"dataset_name": "job1", "order": 1},
            {"dataset_name": "job2", "order": 2},
            {"dataset_name": "job3", "order": 3},
        ]

        for job in jobs:
            fake_redis_server.lpush("job_queue", json.dumps(job))

        # Dequeue and verify order
        dequeued_jobs = []
        for _ in range(3):
            _, job_json = fake_redis_server.brpop("job_queue", timeout=1)
            dequeued_jobs.append(json.loads(job_json))

        assert dequeued_jobs[0]["order"] == 1
        assert dequeued_jobs[1]["order"] == 2
        assert dequeued_jobs[2]["order"] == 3

    def test_empty_queue_timeout(self, fake_redis_server):
        """Test that brpop returns None when queue is empty."""
        # Try to dequeue from empty queue with timeout
        result = fake_redis_server.brpop("job_queue", timeout=1)

        assert result is None

    def test_queue_persistence_across_operations(
        self, fake_redis_server, sample_job_payload
    ):
        """Test that queue maintains state across operations."""
        # Add job
        fake_redis_server.lpush("job_queue", json.dumps(sample_job_payload))

        # Check queue length
        assert fake_redis_server.llen("job_queue") == 1

        # Add another job
        fake_redis_server.lpush("job_queue", json.dumps(sample_job_payload))
        assert fake_redis_server.llen("job_queue") == 2

        # Remove one job
        fake_redis_server.brpop("job_queue", timeout=1)
        assert fake_redis_server.llen("job_queue") == 1


class TestWorkerJobProcessing:
    """Test the worker's job processing logic."""

    @patch("subprocess.run")
    def test_worker_processes_single_job(
        self, mock_subprocess, fake_redis_server, sample_job_payload
    ):
        """Test that worker successfully processes a single job from queue."""
        # Arrange
        mock_subprocess.return_value = Mock(returncode=0)
        job_scheduler = JobScheduler()
        models = ["linear_regression"]

        # Enqueue job
        fake_redis_server.lpush("job_queue", json.dumps(sample_job_payload))

        # Act - Simulate worker loop iteration
        result = fake_redis_server.brpop("job_queue", timeout=1)

        if result:
            _, job_data_json = result
            job_data = json.loads(job_data_json)

            for model_type in models:
                job_scheduler.build_images(job_data, model_type)
                job_scheduler.fill_job_map()

        # Assert
        assert len(job_scheduler.job_map) == 1
        assert fake_redis_server.llen("job_queue") == 0

    @patch("subprocess.run")
    def test_worker_processes_multiple_models(
        self, mock_subprocess, fake_redis_server, sample_job_payload
    ):
        """Test worker processes job for all model types."""
        # Arrange
        mock_subprocess.return_value = Mock(returncode=0)
        job_scheduler = JobScheduler()
        models = ["linear_regression", "random_forest", "xgboost", "feed_forward_nn"]

        # Enqueue job
        fake_redis_server.lpush("job_queue", json.dumps(sample_job_payload))

        # Act
        result = fake_redis_server.brpop("job_queue", timeout=1)

        if result:
            _, job_data_json = result
            job_data = json.loads(job_data_json)

            for model_type in models:
                job_scheduler.build_images(job_data, model_type)
                job_scheduler.fill_job_map()

        # Assert
        # linear_regression: 1
        # random_forest: 3
        # xgboost: 3
        # feed_forward_nn: 3
        # Total: 10
        assert len(job_scheduler.job_map) == 10
        assert mock_subprocess.call_count == 10

    @patch("subprocess.run")
    def test_worker_handles_malformed_json(self, mock_subprocess, fake_redis_server):
        """Test worker handles malformed JSON gracefully."""
        # Arrange
        fake_redis_server.lpush("job_queue", "invalid json {{{")
        job_scheduler = JobScheduler()

        # Act & Assert
        result = fake_redis_server.brpop("job_queue", timeout=1)

        if result:
            _, job_data_json = result
            with pytest.raises(json.JSONDecodeError):
                json.loads(job_data_json)

    @patch("subprocess.run")
    def test_worker_continues_after_job_failure(
        self, mock_subprocess, fake_redis_server
    ):
        """Test that worker continues processing after a job fails."""
        # Arrange
        job_scheduler = JobScheduler()

        # First job will fail
        job1 = {
            "dataset_name": "job1",
            "dataset_path": "/invalid",
            "target_name": "y",
            "task_type": "classification",
        }
        # Second job will succeed
        job2 = {
            "dataset_name": "job2",
            "dataset_path": "/valid",
            "target_name": "y",
            "task_type": "classification",
        }

        fake_redis_server.lpush("job_queue", json.dumps(job1))
        fake_redis_server.lpush("job_queue", json.dumps(job2))

        # Mock subprocess to fail first, succeed second
        mock_subprocess.side_effect = [
            subprocess.CalledProcessError(1, "docker build"),
            Mock(returncode=0),
        ]

        # Act - Process first job (fails)
        result = fake_redis_server.brpop("job_queue", timeout=1)
        if result:
            _, job_data_json = result
            job_data = json.loads(job_data_json)
            try:
                job_scheduler.build_images(job_data, "linear_regression")
            except subprocess.CalledProcessError:
                pass  # Expected failure

        # Process second job (succeeds)
        result = fake_redis_server.brpop("job_queue", timeout=1)
        if result:
            _, job_data_json = result
            job_data = json.loads(job_data_json)
            job_scheduler.build_images(job_data, "linear_regression")
            job_scheduler.fill_job_map()

        # Assert second job was processed successfully
        assert len(job_scheduler.job_map) == 1


class TestEndToEndWorkflow:
    """End-to-end tests simulating complete workflow."""

    @patch("subprocess.run")
    @patch("builtins.open")
    @patch("pathlib.Path.mkdir")
    def test_complete_workflow_simulation(
        self, mock_mkdir, mock_open, mock_subprocess, fake_redis_server
    ):
        """Simulate complete workflow from API to worker processing."""
        # Arrange
        mock_subprocess.return_value = Mock(returncode=0)
        job_scheduler = JobScheduler()

        # Step 1: API receives dataset and enqueues job (simulated)
        api_request = {
            "name": "iris",
            "target_name": "species",
            "task_type": "classification",
            "dataset_csv": "data,here\n1,2\n",
        }

        job_data = {
            "dataset_path": f"/data_storage/{api_request['name']}.csv",
            "dataset_name": api_request["name"],
            "target_name": api_request["target_name"],
            "task_type": api_request["task_type"],
        }

        fake_redis_server.lpush("job_queue", json.dumps(job_data))

        # Step 2: Worker processes job
        result = fake_redis_server.brpop("job_queue", timeout=1)

        processed_jobs = []
        if result:
            _, job_data_json = result
            job_data = json.loads(job_data_json)

            for model in ["linear_regression", "random_forest"]:
                job_scheduler.build_images(job_data, model)
                job_scheduler.fill_job_map()
                processed_jobs.append(model)

        # Assert
        assert len(processed_jobs) == 2
        # linear_regression: 1
        # random_forest: 3
        # Total: 4
        assert len(job_scheduler.job_map) == 4
        assert fake_redis_server.llen("job_queue") == 0

    def test_concurrent_job_enqueueing(self, fake_redis_server):
        """Test multiple jobs can be enqueued concurrently."""
        # Simulate multiple API requests
        jobs = []
        for i in range(10):
            job = {
                "dataset_name": f"dataset_{i}",
                "dataset_path": f"/path/to/dataset_{i}.csv",
                "target_name": "target",
                "task_type": "classification",
            }
            jobs.append(job)
            fake_redis_server.lpush("job_queue", json.dumps(job))

        # Assert all jobs are in queue
        assert fake_redis_server.llen("job_queue") == 10

        # Process all jobs
        processed = []
        for _ in range(10):
            result = fake_redis_server.brpop("job_queue", timeout=1)
            if result:
                _, job_json = result
                processed.append(json.loads(job_json))

        assert len(processed) == 10
        assert fake_redis_server.llen("job_queue") == 0


# Import subprocess for the error test
import subprocess
