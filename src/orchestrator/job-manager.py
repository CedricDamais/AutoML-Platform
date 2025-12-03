from utils.logging import logger
from pathlib import Path

from exceptions import (
    InternalServerError,
    NotFoundError,
)

import pandas as pd

class JobManager:
    def __init__(self, dataset_path: Path):
        logger.info("Initializing Job Manager")
        self.dataset : Path = dataset_path
        self.pd_df : pd.DataFrame = self.load_dataset()
        logger.info("Job Manager initialized successfully")
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load dataset from the given path.
        
        :param self: instance of the Job Manager
        :return: Loaded dataset as a DataFrame
        :rtype: pd.DataFrame
        """
        logger.info(f"Loading dataset from {self.dataset}")
        try:
            if self.dataset.suffix == ".csv":
                df = pd.read_csv(self.dataset)
            elif self.dataset.suffix in [".xls", ".xlsx"]:
                df = pd.read_excel(self.dataset)
            elif self.dataset.suffix == ".json":
                df = pd.read_json(self.dataset)
            else:
                raise ValueError("Unsupported file format. Please use CSV, Excel, or JSON files.")
            logger.info("Dataset loaded successfully")
            return df
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise InternalServerError() from e