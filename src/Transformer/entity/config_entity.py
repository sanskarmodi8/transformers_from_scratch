from dataclasses import dataclass
from pathlib import Path

from box import ConfigBox

# entity classes for final configuration of each component of the pipeline


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    lang_pair: str
    data_path: Path
