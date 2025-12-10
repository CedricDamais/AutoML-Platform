import os
import mlflow
from typing import Optional
import logging


def get_best_model_path(
    mlruns_path: str = "./mlruns",
    experiment_ids: list[str] = None,
    metric_name: str = "test_f1",
) -> tuple[Optional[str], Optional[str]]:

    """
    Finds the best model from MLflow runs based on a specific metric.
    
    Args:
        mlruns_path: Path to the mlruns directory.
        experiment_ids: List of experiment IDs to search. If None, searches all known experiments.
        metric_name: The metric to optimize (assumes higher is better).
        
    Returns:
        Tuple containing (Absolute path to the model artifact, run_id).
        Returns (None, None) if no model is found.
    """
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_path)}")
    
    if experiment_ids is None:
        try:
            experiments = mlflow.search_experiments()
            experiment_ids = [exp.experiment_id for exp in experiments]
        except Exception:
            experiment_ids = []

        if os.path.exists(mlruns_path):
             for name in os.listdir(mlruns_path):
                 if name.isdigit() and os.path.isdir(os.path.join(mlruns_path, name)):
                     if name not in experiment_ids:
                         experiment_ids.append(name)
            
    best_run_id = None
    best_metric_value = -float("inf")
    
    logging.info(f"Searching for best model in experiments: {experiment_ids} based on {metric_name}...")
    
    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
    )
    
    if runs.empty:
        logging.warning("No runs found (empty dataframe).")
        if experiment_ids:
             return None, None
        return None, None

    # Sort runs by metric in descending order
    # Check if metric exists in columns
    metric_col = f"metrics.{metric_name}"
    if metric_col in runs.columns:
        sorted_runs = runs.sort_values(by=[metric_col, "start_time"], ascending=[False, False])
        
        best_run_id = None
        best_run = None
        artifact_path = None
        
        for _, run in sorted_runs.iterrows():
            uri = run.artifact_uri
            
            if uri.startswith("file://"):
                path = uri[7:]
            elif uri.startswith("mlflow-artifacts:"):
                 parts = uri.split(":")
                 if len(parts) > 1:
                     path_part = parts[1]
                     candidate = os.path.join(mlruns_path, path_part.strip("/"))
                     if os.path.exists(candidate):
                         path = candidate
                     else:
                         path = path_part
                 else:
                     path = uri
            else:
                path = uri
            
            logging.info(f"Checking candidate run {run.run_id} (acc={run[metric_col]})... Path: {path}")
            
            has_model = False
            for root, dirs, files in os.walk(path):
                if "MLmodel" in files:
                    has_model = True
                    artifact_path = root
                    break
            
            if not has_model:
                 try:
                     exp_id = run["experiment_id"]
                     run_id = run["run_id"]
                     
                     outputs_dir = os.path.join(mlruns_path, exp_id, run_id, "outputs")
                     if os.path.exists(outputs_dir):
                         for item in os.listdir(outputs_dir):
                             if item.startswith("m-"):
                                 model_id = item
                                 root_dir = os.path.dirname(os.path.abspath(mlruns_path))
                                 alt_path = os.path.join(root_dir, "mlartifacts", exp_id, "models", model_id, "artifacts")
                                 
                                 if os.path.exists(os.path.join(alt_path, "MLmodel")):
                                     logging.info(f"Found MLmodel in custom path: {alt_path}")
                                     has_model = True
                                     artifact_path = alt_path
                                     break
                 except Exception as e:
                     logging.warning(f"Failed to check custom artifact path: {e}")

            if has_model:

                best_run_id = run.run_id
                best_metric_value = run[metric_col]
                logging.info(f"Found valid best run: {best_run_id} with {metric_name}: {best_metric_value}")
                return artifact_path, best_run_id
            else:
                logging.warning(f"Run {run.run_id} skipped: No 'MLmodel' file found in artifacts.")

        if best_run is not None:
             logging.info(f"Found valid best run: {best_run_id} with {metric_name}: {best_metric_value}")
             return artifact_path, best_run_id
        
        logging.warning("No valid model found via MLflow run metadata.")
        
        if experiment_ids and len(experiment_ids) == 1:
            logging.error("Strict experiment targeting enabled. Skipping filesystem fallback.")
            return None, None

        logging.info("Attempting fallback filesystem search...")
        
        latest_model_path = None
        latest_time = 0
        
        for root, dirs, files in os.walk(mlruns_path):
            if "MLmodel" in files:
                full_path = os.path.join(root, "MLmodel")
                try:
                    mtime = os.path.getmtime(full_path)
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_model_path = root
                except OSError:
                    continue
                    
        if latest_model_path:
             logging.info(f"Fallback: Found latest model at {latest_model_path}")
             return latest_model_path, None
             
        return None

    else:
        logging.warning(f"Metric {metric_name} not found in runs columns. Trying fallback to test_accuracy.")
        if metric_name != "test_accuracy":
            return get_best_model_path(mlruns_path, experiment_ids, "test_accuracy")
        
        # If fallback fails, return None (don't go to filesystem fallback)
        return None, None




if __name__ == "__main__":
    path, run_id = get_best_model_path()
    if path:
        print(f"Best model path: {path} (Run ID: {run_id})")
    else:
        print("Model not found.")
