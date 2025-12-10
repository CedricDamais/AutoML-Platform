import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import subprocess
from src.deployment.model_selector import get_best_model_path
from src.kubernetes.k3s_builder import (
    K3sProject,
    K3sNamespace,
    K3sKustomization,
    K3sModule,
    K3sDeployment,
    K3sService,
    K3sIngress,
    run_k3s_project,
)
import logging
import shutil
import time
import signal


logging.basicConfig(
    filename='deployment.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())

def build_docker_image(image_name: str, dockerfile_path: str, minikube_mode: bool = False):
    logging.info(f"Building Docker image {image_name}...")
    
    if minikube_mode:
        logging.info("Using Minikube image build...")
        # Minikube image build command
        cmd = [
            "minikube", "image", "build",
            "-t", image_name,
            "-f", dockerfile_path,
            "."
        ]
    else:
        cmd = [
            "docker", "build",
            "-f", dockerfile_path,
            "-t", image_name,
            "."
        ]

    try:
        subprocess.check_call(cmd)
        logging.info("Docker image built successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to build image: {e}")
        raise

    if not minikube_mode:
        logging.info(f"NOTE: If using local K3s, ensure '{image_name}' is available to the cluster.")
        logging.info(f"      e.g., 'k3s ctr images import {image_name}.tar' or use a registry.")

def start_port_forwarding(
    namespace: str,
    service_name: str,
    local_port: int = 8000,
    remote_port: int = 8000

):
    """
    Starts kubectl port-forward in a detached subprocess.
    Kills any existing process on the local port first (simple heuristic).
    """
    logging.info(f"Setting up port forwarding for {service_name} on port {local_port}...")
    
    # 1. Attempt to kill existing port-forward on this port
    # This is a bit aggressive but necessary for a dev tool.
    try:
        # Find pids listening on local_port
        pid_bytes = subprocess.check_output(["lsof", "-ti", f":{local_port}"], stderr=subprocess.DEVNULL)
        pids = pid_bytes.decode().strip().split('\n')
        for pid in pids:
            if pid:
                logging.info(f"Killing existing process on port {local_port} (PID: {pid})")
                os.kill(int(pid), signal.SIGTERM)
        time.sleep(1)
    except Exception:
        pass

    # 2. Start new port-forward using nohup to ensure it survives and handles logging
    cmd_str = f"nohup kubectl port-forward -n {namespace} svc/{service_name} {local_port}:{remote_port} > port_forward.log 2>&1 &"
    
    try:
        subprocess.Popen(
            cmd_str,
            shell=True,
            preexec_fn=os.setpgrp
        )
        logging.info(f"Port forwarding started in background. Access at http://localhost:{local_port}")
    except Exception as e:
        logging.error(f"Failed to start port forwarding: {e}")



def deploy(
    image_name: str = "ml-inference:latest",
    namespace_name: str = "ml-deployment",
    port: int = 8000,

    minikube: bool = False,
    experiment_id: str = None,
    status_callback: callable = None
):
    def report(msg, step=None):
        logging.info(msg)
        if status_callback:
            if step:
                status_callback(msg, step=step)
            else:
                status_callback(msg)

    report(f"Selecting best model (experiment_id={experiment_id})...", step="SELECT_MODEL")
    experiment_ids = [experiment_id] if experiment_id else None
    model_path, run_id = get_best_model_path(experiment_ids=experiment_ids)
    if not model_path:

        logging.error("No suitable model found in mlruns. Aborting deployment.")
        return

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isabs(model_path):
        model_source_path = os.path.join(project_root, model_path)
    else:
        model_source_path = model_path

    build_artifacts_dir = os.path.join(project_root, "model_build_artifacts")
    if os.path.exists(build_artifacts_dir):
        shutil.rmtree(build_artifacts_dir)
    
    try:
        report(f"Copying model from {model_source_path} to {build_artifacts_dir}...")
        shutil.copytree(model_source_path, build_artifacts_dir)
    except Exception as e:
        status_callback(f"Error: {e}") if status_callback else None
        logging.error(f"Failed to copy model artifacts: {e}")
        return

    try:
        report(f"Building Docker image {image_name}...", step="BUILD_IMAGE")
        build_docker_image(image_name, "src/deployment/docker/Dockerfile.inference", minikube_mode=minikube)
    finally:
        pass

    namespace = K3sNamespace(name=namespace_name)
    
    project_kustomization = K3sKustomization(
        namespace=namespace, resources=["namespace.yml", "inference-service"]
    )
    deployment = K3sDeployment(
        namespace=namespace,
        name="inference-service",
        docker_image=image_name,
        replicas=1,
        container_port=8000,
        model_path="/app/model", # Point to internal path in container
        env_vars={"MODEL_PATH": "/app/model"}
    )

    service = K3sService(
        namespace=namespace,
        name="inference-service",
        port=8000,
        target_port=8000,
        service_type="NodePort"
    )

    ingress = K3sIngress(
        namespace=namespace,
        name="inference-ingress",
        service_name="inference-service",
        service_port=8000,
        host="model.example.com"
    )


    module = K3sModule(
        namespace=namespace,
        name="inference-service",
        kustomization=K3sKustomization(
            namespace=namespace, 
            resources=["deployment.yml", "service.yml", "ingress.yml"]
        ),
        deployment=deployment,
        service=service,
        ingress=ingress
    )

    project = K3sProject(
        namespace=namespace,
        kustomization=project_kustomization,
        modules=[module]
    )

    report("Applying K3s manifests...", step="APPLY_MANIFESTS")
    try:
        run_k3s_project(project)
        
        # Force restart to pick up the new image (since tag is 'latest'), basically forcing a new deployment
        report("Triggering rollout restart to ensure new image is picked up...", step="ROLLOUT_RESTART")
        subprocess.run(
            ["kubectl", "rollout", "restart", "deployment/inference-service", "-n", namespace_name],
            check=True
        )
        
        report("Waiting for rollout to complete (this may take a minute)...", step="WAIT_ROLLOUT")
        subprocess.run(
            ["kubectl", "rollout", "status", "deployment/inference-service", "-n", namespace_name],
            check=True
        )

        
        report("Deployment applied successfully.")

        logging.info(f"Model deployed in namespace '{namespace_name}'.")
        logging.info(f"Access via Ingress host: {ingress.host}")
    except Exception as e:
        logging.error(f"Failed to apply deployment: {e}")
    
    return run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy best ML model to K3s")
    parser.add_argument("--image", default="ml-inference:latest", help="Docker image name")
    parser.add_argument("--namespace", default="ml-deployment", help="K8s namespace")
    parser.add_argument("--minikube", action="store_true", help="Build image inside Minikube")
    args = parser.parse_args()

    deploy(image_name=args.image, namespace_name=args.namespace, minikube=args.minikube)
