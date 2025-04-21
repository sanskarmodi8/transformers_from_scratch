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


@dataclass(frozen=True)
class BuildModelConfig:
    root_dir: Path
    model_path: Path
    num_layers: int
    d_model: int
    num_heads: int
    dff: int
    vocab_size: int
    dropout: float
    max_length: int


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    trained_model_path: Path
    wandb_project_name: str
    wandb_run_name: str
    warmup_steps: int
    src_tokens_per_batch: int
    tgt_tokens_per_batch: int
    total_steps: int
    lr_factor: float
    clip_grad: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    label_smoothing: float
    last_n_checkpoints_to_avg: int
    checkpoint_interval_minutes: int
    beam_size: int
    d_model: int
    length_penalty: float
    num_layers: int
    vocab_size: int
    dropout: float
    max_length: int
    num_heads: int
    dff: int
