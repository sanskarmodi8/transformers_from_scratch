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
    vocab_size: int
    tokenizer_path: Path
    max_length: int


@dataclass(frozen=True)
class BuildModelConfig:
    root_dir: Path
    model_path: Path
    num_layers: int
    d_model: int
    num_heads: int
    dff: int
    vocab_size: int
