from dataclasses import dataclass
from pathlib import Path

from box import ConfigBox

# entity classes for final configuration of each component of the pipeline


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    data_path: Path
    preprocessed_data_path: Path
