from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
import yaml


@dataclass
class K3sJob:
    namespace: K3sNamespace
    name: str
    docker_image: str
    env_vars: dict[str, str] = None


@dataclass
class K3sKustomization:
    namespace: K3sNamespace
    resources: list[K3sModule | str]

@dataclass
class K3sDeployment:
    namespace: K3sNamespace
    name: str
    docker_image: str
    replicas: int = 1
    container_port: int = 8000
    env_vars: dict[str, str] = None
    model_path: str = None

@dataclass
class K3sService:
    namespace: K3sNamespace
    name: str
    port: int
    target_port: int
    service_type: str = "ClusterIP"

@dataclass
class K3sIngress:
    namespace: K3sNamespace
    name: str
    service_name: str
    service_port: int
    host: str


@dataclass
class K3sModule:
    namespace: K3sNamespace
    name: str
    kustomization: K3sKustomization
    job: K3sJob = None
    deployment: K3sDeployment = None
    service: K3sService = None
    ingress: K3sIngress = None

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


def create_k3s_project(
    name: str,
    modules: dict[str, str],
    env_vars: dict[str, str] = None,
) -> K3sProject:
    namespace = K3sNamespace(name=name)
    kustomization = K3sKustomization(
        namespace=namespace, resources=["namespace.yml"] + list(modules.keys())
    )
    module_list = [
        K3sModule(
            namespace=namespace,
            name=module_name,
            job=K3sJob(
                namespace=namespace,
                name=module_name,
                docker_image=docker_image,
                env_vars=env_vars,
            ),
            kustomization=K3sKustomization(namespace=namespace, resources=["job.yml"]),
        )
        for i, (module_name, docker_image) in enumerate(modules.items())
    ]

    return K3sProject(
        namespace=namespace, kustomization=kustomization, modules=module_list
    )


def k3s_job_to_yaml(job: K3sJob, **kwargs) -> str:
    labels = {
        "app": job.namespace.name,
        "component": job.name,
    }
    metadata = {
        "name": job.name,
        "labels": labels,
    }
    container_spec = {
        "name": job.name,
        "image": job.docker_image,
        "imagePullPolicy": "Never",
        "env": [
            {
                "name": "MLFLOW_TRACKING_URI",
                "value": f"http://{os.environ.get('IP_ADDR', '127.0.0.1')}:5001",
            }
        ],
        "resources": {
            "requests": {"memory": "512Mi", "cpu": "500m"},
            "limits": {"memory": "2Gi", "cpu": "1000m"},
        },
    }

    if job.env_vars:
        for key, value in job.env_vars.items():
            container_spec["env"].append({"name": key, "value": str(value)})

    return yaml.dump(
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": metadata,
            "spec": {
                "backoffLimit": 3,
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [container_spec],
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

        with open(f"{module_path}/kustomization.yml", "w") as file:
            k3s_kustomization_to_yaml(
                module.kustomization, stream=file, default_flow_style=False
            )
        
        if module.job:
            with open(f"{module_path}/job.yml", "w") as file:
                k3s_job_to_yaml(module.job, stream=file, default_flow_style=False)
        
        if module.deployment:
            with open(f"{module_path}/deployment.yml", "w") as file:
                k3s_deployment_to_yaml(module.deployment, stream=file, default_flow_style=False)
        
        if module.service:
            with open(f"{module_path}/service.yml", "w") as file:
                k3s_service_to_yaml(module.service, stream=file, default_flow_style=False)
        
        if module.ingress:
             with open(f"{module_path}/ingress.yml", "w") as file:
                k3s_ingress_to_yaml(module.ingress, stream=file, default_flow_style=False)

def k3s_deployment_to_yaml(deployment: K3sDeployment, **kwargs) -> str:
    labels = {
        "app": deployment.namespace.name,
        "component": deployment.name,
    }
    
    container_spec = {
        "name": deployment.name,
        "image": deployment.docker_image,
        "imagePullPolicy": "Never",
        "ports": [{"containerPort": deployment.container_port}],
        "resources": {
            "requests": {"memory": "512Mi", "cpu": "500m"},
            "limits": {"memory": "2Gi", "cpu": "1000m"},
        },
    }

    env_list = []
    if deployment.env_vars:
        for key, value in deployment.env_vars.items():
            env_list.append({"name": key, "value": str(value)})
            
    if deployment.model_path:
        env_list.append({"name": "MODEL_PATH", "value": deployment.model_path})
        
    if env_list:
        container_spec["env"] = env_list

    return yaml.dump(
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": deployment.name, "namespace": deployment.namespace.name, "labels": labels},
            "spec": {
                "replicas": deployment.replicas,
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "containers": [container_spec],
                    },
                },
            },
        },
        **kwargs,
    )

def k3s_service_to_yaml(service: K3sService, **kwargs) -> str:
    return yaml.dump(
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": service.name, "namespace": service.namespace.name},
            "spec": {
                "type": service.service_type,
                "selector": {
                    "app": service.namespace.name,
                    "component": service.name,
                },
                "ports": [
                    {
                        "protocol": "TCP",
                        "port": service.port,
                        "targetPort": service.target_port,
                    }
                ],
            },
        },
        **kwargs,
    )

def k3s_ingress_to_yaml(ingress: K3sIngress, **kwargs) -> str:
    return yaml.dump(
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {"name": ingress.name, "namespace": ingress.namespace.name},
            "spec": {
                "rules": [
                    {
                        "host": ingress.host,
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": ingress.service_name,
                                            "port": {"number": ingress.service_port},
                                        }
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
        },
        **kwargs,
    )

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
