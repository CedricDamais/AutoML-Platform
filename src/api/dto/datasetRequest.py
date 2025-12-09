from pydantic import BaseModel, Field
from typing import Optional, Literal

class DatasetRequest(BaseModel):
    name: str = Field(..., description="Dataset name")
    target_name: str = Field(..., description="Data to predict")
    task_type: Optional[Literal["classification", "regression"]] = Field("classification", description="ML Problem to solve")
    dataset_csv : Optional[str] = Field(None, description="Small Dataset only (< 10MB)")
    mlflow_experiment: Optional[str] = Field("automl-experiments", description="MLflow experiment name to log runs into")
