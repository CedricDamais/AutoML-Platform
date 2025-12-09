import pytest
import yaml
from src.kubernetes.k3s_builder import (
    create_k3s_project,
    k3s_job_to_yaml,
    k3s_kustomization_to_yaml,
    k3s_namespace_to_yaml,
    K3sJob,
    K3sKustomization,
    K3sModule,
    K3sNamespace,
    K3sProject,
)


@pytest.fixture
def sample_namespace():
    return K3sNamespace(name="test-app")


@pytest.fixture
def sample_job(sample_namespace):
    return K3sJob(
        namespace=sample_namespace,
        name="web-server",
        docker_image="nginx:latest",
    )


@pytest.fixture
def sample_kustomization(sample_namespace):
    return K3sKustomization(namespace=sample_namespace, resources=["job.yml"])


def test_k3s_module_str_representation(
    sample_job, sample_kustomization, sample_namespace
):
    """Test the __str__ method of K3sModule."""
    module = K3sModule(
        namespace=sample_namespace,
        name="my-module",
        job=sample_job,
        kustomization=sample_kustomization,
    )
    assert str(module) == "my-module"


def test_create_k3s_project_structure():
    """Test that create_k3s_project builds the correct object hierarchy."""
    modules_config = {"frontend": "react:18", "backend": "python:3.9"}
    project = create_k3s_project(name="my-full-stack", modules=modules_config)

    # Check Project and Namespace
    assert isinstance(project, K3sProject)
    assert project.namespace.name == "my-full-stack"

    # Check Root Kustomization
    assert project.kustomization.namespace.name == "my-full-stack"
    assert "frontend" in project.kustomization.resources
    assert "backend" in project.kustomization.resources

    # Check Modules list
    assert len(project.modules) == 2

    # Verify module details
    frontend = next(m for m in project.modules if m.name == "frontend")
    assert frontend.job.docker_image == "react:18"
    assert frontend.job.name == "frontend"


def test_create_k3s_project_multiple_jobs():
    """Test that multiple jobs are created correctly."""
    modules_config = {"service-a": "img-a", "service-b": "img-b", "service-c": "img-c"}
    project = create_k3s_project(name="job-test", modules=modules_config)

    # Check that all jobs are created
    assert len(project.modules) == 3

    # Check that each module has a job with correct image
    job_names = {m.job.name for m in project.modules}
    job_images = {m.job.docker_image for m in project.modules}

    assert job_names == {"service-a", "service-b", "service-c"}
    assert job_images == {"img-a", "img-b", "img-c"}


# --- YAML Generation Tests ---


def test_k3s_namespace_to_yaml(sample_namespace):
    """Test namespace YAML generation."""
    yaml_output = k3s_namespace_to_yaml(sample_namespace)
    data = yaml.safe_load(yaml_output)

    assert data["apiVersion"] == "v1"
    assert data["kind"] == "Namespace"
    assert data["metadata"]["name"] == "test-app"


def test_k3s_kustomization_to_yaml(sample_kustomization):
    """Test kustomization YAML generation."""
    yaml_output = k3s_kustomization_to_yaml(sample_kustomization)
    data = yaml.safe_load(yaml_output)

    assert data["kind"] == "Kustomization"
    assert data["namespace"] == "test-app"
    assert data["resources"] == ["job.yml"]


def test_k3s_job_to_yaml(sample_job):
    """Test job YAML generation."""
    yaml_output = k3s_job_to_yaml(sample_job)
    data = yaml.safe_load(yaml_output)

    assert data["apiVersion"] == "batch/v1"
    assert data["kind"] == "Job"

    # Check Metadata
    assert data["metadata"]["name"] == "web-server"
    assert data["metadata"]["labels"]["app"] == "test-app"
    assert data["metadata"]["labels"]["component"] == "web-server"

    # Check Spec
    spec = data["spec"]
    assert spec["backoffLimit"] == 3

    # Check Template
    template_spec = spec["template"]["spec"]
    assert template_spec["restartPolicy"] == "Never"

    # Check Container
    container = template_spec["containers"][0]
    assert container["name"] == "web-server"
    assert container["image"] == "nginx:latest"
