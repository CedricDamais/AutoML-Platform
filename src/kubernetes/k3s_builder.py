from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
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
    name: str, modules: dict[str, str], start_port: int = 8081
) -> K3sProject:
    namespace = K3sNamespace(name=name)
    kustomization = K3sKustomization(
        namespace=namespace, resources=["namespace.yml"] + list(modules.keys())
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
            kustomization=K3sKustomization(
                namespace=namespace, resources=["deployment.yml", "service.yml"]
            ),
            service=K3sService(
                namespace=namespace, name=module_name, port=start_port + i
            ),
        )
        for i, (module_name, docker_image) in enumerate(modules.items())
    ]

    return K3sProject(
        namespace=namespace, kustomization=kustomization, modules=module_list
    )


def k3s_deployment_to_yaml(deployment: K3sDeployment, **kwargs) -> str:
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
        },
        **kwargs,
    )


def k3s_kustomization_to_yaml(kustomization: K3sKustomization, **kwargs) -> str:
    return yaml.dump(
        {
            "kind": "Kustomization",
            "namespace": kustomization.namespace.name,
            "resources": kustomization.resources,
        },
        **kwargs,
    )


def k3s_namespace_to_yaml(namespace: K3sNamespace, **kwargs) -> str:
    return yaml.dump(
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": namespace.name,
            },
        },
        **kwargs,
    )


def k3s_service_to_yaml(service: K3sService, **kwargs) -> str:
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
        },
        **kwargs,
    )


def dump_k3s_project(project: K3sProject, path: str = "/tmp/k3s_projects/"):
    path += "/" + project.namespace.name

    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

    with open(f"{path}/namespace.yml", "w") as file:
        k3s_namespace_to_yaml(project.namespace, stream=file, default_flow_style=False)

    with open(f"{path}/kustomization.yml", "w") as file:
        k3s_kustomization_to_yaml(
            project.kustomization, stream=file, default_flow_style=False
        )

    for module in project.modules:
        module_path = f"{path}/{module.name}"
        os.mkdir(module_path)

        with open(f"{module_path}/deployment.yml", "w") as file:
            k3s_deployment_to_yaml(
                module.deployment, stream=file, default_flow_style=False
            )

        with open(f"{module_path}/kustomization.yml", "w") as file:
            k3s_kustomization_to_yaml(
                module.kustomization, stream=file, default_flow_style=False
            )

        with open(f"{module_path}/service.yml", "w") as file:
            k3s_service_to_yaml(module.service, stream=file, default_flow_style=False)


def run_k3s_project(project: K3sProject):
    path = "/tmp/k3s_projects"
    dump_k3s_project(project=project, path=path)
    subprocess.run(
        [
            "kubectl",
            "apply",
            "-n",
            project.namespace.name,
            "-k",
            f"{path}/{project.namespace.name}",
        ],
        check=True,
    )
