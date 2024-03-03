"""DataPrep Placeholder"""
from dataclasses import dataclass
from typing import List


@dataclass
class DataPrepConfig:
    """
     Attributes:
        columns (list): List of columns to subset to during data preprocessing
        drop_missing (bool): Flag to indicate whether or not to drop missing values
    """
    columns: List[str]
    drop_missing: bool = True

# Add example DataPrep class dropping null val