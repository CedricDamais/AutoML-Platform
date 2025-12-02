from __future__ import annotations

from dataclasses import dataclass
import yaml


@dataclass
class K3sDeployment:
    namespace: K3sNamespace
    name: str
    docker_image: str
    port: int


@dataclass
class K3sKustomization:
    namespace: K3sNamespace
    resources: list[K3sModule | str]


@dataclass
class K3sModule:
    namespace: K3sNamespace
    name: str
    deployment: K3sDeployment
    kustomization: K3sKustomization
    service: K3sService

    def __str__(self):
        return self.name


@dataclass
class K3sNamespace:
    name: str


@dataclass
class K3sProject:
    namespace: K3sNamespace
    kustomization: K3sKustomization
    modules: list[K3sModule]


@dataclass
class K3sService:
    namespace: K3sNamespace
    name: str
    port: int


def create_k3s_project(
    name: str, modules: dict[str, str], start_port: int = 8000
) -> K3sProject:
    namespace = K3sNamespace(name=name)
    kustomization = K3sKustomization(
        namespace=namespace, resources=list(modules.keys())
    )
    module_list = [
        K3sModule(
            namespace=namespace,
            name=module_name,
            deployment=K3sDeployment(
                namespace=namespace,
                name=module_name,
                docker_image=docker_image,
                port=start_port + i,
            ),
            kustomization=["deployment.yml", "service.yml"],
            service=K3sService(
                namespace=namespace, name=module_name, port=start_port + i
            ),
        )
        for i, (module_name, docker_image) in enumerate(modules.items())
    ]

    return K3sProject(
        namespace=namespace, kustomization=kustomization, modules=module_list
    )


def k3s_deployment_to_yaml(deployment: K3sDeployment) -> str:
    labels = {
        "app": deployment.namespace.name,
        "component": deployment.name,
    }
    metadata = {
        "name": deployment.name,
        "labels": labels,
    }
    return yaml.dump(
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": metadata,
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": labels,
                },
                "template": {
                    "metadata": metadata,
                    "spec": {
                        "containers": [
                            {
                                "name": deployment.name,
                                "image": deployment.docker_image,
                                "ports": [
                                    {
                                        "name": "http",
                                        "containerPort": deployment.port,
                                    }
                                ],
                            }
                        ]
                    },
                },
            },
        }
    )


def k3s_kustomization_to_yaml(kustomization: K3sKustomization) -> str:
    return yaml.dump(
        {
            "kind": "Kustomization",
            "namespace": kustomization.namespace.name,
            "resources": kustomization.resources,
        }
    )


def k3s_namespace_to_yaml(namespace: K3sNamespace) -> str:
    return yaml.dump(
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": namespace.name,
            },
        }
    )


def k3s_service_to_yaml(service: K3sService) -> str:
    return yaml.dump(
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": service.name,
            },
            "spec": {
                "selector": {
                    "app": service.namespace.name,
                    "component": service.name,
                },
                "ports": [
                    {
                        "name": "http",
                        "port": service.port,
                        "targetPort": service.port,
                    }
                ],
            },
        }
    )
