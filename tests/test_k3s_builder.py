import pytest
import yaml
from src.kubernetes.k3s_builder import (
    create_k3s_project,
    k3s_deployment_to_yaml,
    k3s_kustomization_to_yaml,
    k3s_namespace_to_yaml,
    k3s_service_to_yaml,
    K3sDeployment,
    K3sKustomization,
    K3sModule,
    K3sNamespace,
    K3sProject,
    K3sService,
)


@pytest.fixture
def sample_namespace():
    return K3sNamespace(name="test-app")


@pytest.fixture
def sample_deployment(sample_namespace):
    return K3sDeployment(
        namespace=sample_namespace,
        name="web-server",
        docker_image="nginx:latest",
        port=8080,
    )


@pytest.fixture
def sample_service(sample_namespace):
    return K3sService(namespace=sample_namespace, name="web-server", port=8080)


@pytest.fixture
def sample_kustomization(sample_namespace):
    return K3sKustomization(
        namespace=sample_namespace, resources=["deployment.yml", "service.yml"]
    )


def test_k3s_module_str_representation(
    sample_deployment, sample_service, sample_kustomization, sample_namespace
):
    """Test the __str__ method of K3sModule."""
    module = K3sModule(
        namespace=sample_namespace,
        name="my-module",
        deployment=sample_deployment,
        kustomization=sample_kustomization,
        service=sample_service,
    )
    assert str(module) == "my-module"


def test_create_k3s_project_structure():
    """Test that create_k3s_project builds the correct object hierarchy."""
    modules_config = {"frontend": "react:18", "backend": "python:3.9"}
    project = create_k3s_project(
        name="my-full-stack", modules=modules_config, start_port=3000
    )

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
    assert frontend.deployment.docker_image == "react:18"
    assert frontend.deployment.name == "frontend"


def test_create_k3s_project_port_increment():
    """Test that ports increment correctly for each module."""
    modules_config = {"service-a": "img-a", "service-b": "img-b", "service-c": "img-c"}
    start_port = 9000
    project = create_k3s_project(
        name="port-test", modules=modules_config, start_port=start_port
    )

    # Sort modules by name to ensure consistent checking order if dict order varies
    # (though dicts are ordered in modern Python, logic inside function uses enumerate)
    # We just need to check that unique ports are assigned starting from 9000.

    ports = [m.deployment.port for m in project.modules]
    expected_ports = {9000, 9001, 9002}

    assert set(ports) == expected_ports

    # Check that service port matches deployment port
    for module in project.modules:
        assert module.service.port == module.deployment.port


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
    assert data["resources"] == ["deployment.yml", "service.yml"]


def test_k3s_deployment_to_yaml(sample_deployment):
    """Test deployment YAML generation."""
    yaml_output = k3s_deployment_to_yaml(sample_deployment)
    data = yaml.safe_load(yaml_output)

    assert data["apiVersion"] == "apps/v1"
    assert data["kind"] == "Deployment"

    # Check Metadata
    assert data["metadata"]["name"] == "web-server"
    assert data["metadata"]["labels"]["app"] == "test-app"
    assert data["metadata"]["labels"]["component"] == "web-server"

    # Check Spec
    spec = data["spec"]
    assert spec["replicas"] == 1
    assert spec["selector"]["matchLabels"]["component"] == "web-server"

    # Check Container
    container = spec["template"]["spec"]["containers"][0]
    assert container["name"] == "web-server"
    assert container["image"] == "nginx:latest"
    assert container["ports"][0]["containerPort"] == 8080


def test_k3s_service_to_yaml(sample_service):
    """Test service YAML generation."""
    yaml_output = k3s_service_to_yaml(sample_service)
    data = yaml.safe_load(yaml_output)

    assert data["apiVersion"] == "v1"
    assert data["kind"] == "Service"
    assert data["metadata"]["name"] == "web-server"

    # Check Selector
    assert data["spec"]["selector"]["app"] == "test-app"
    assert data["spec"]["selector"]["component"] == "web-server"

    # Check Ports
    ports = data["spec"]["ports"]
    assert isinstance(ports, list)
    port = ports[0]
    assert port["name"] == "http"
    assert port["port"] == 8080
    assert port["targetPort"] == 8080
