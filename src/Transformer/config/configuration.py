from pathlib import Path

from Transformer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from Transformer.entity.config_entity import (
    BuildModelConfig,
    DataIngestionConfig,
    DataPreprocessingConfig,
    ModelTrainingConfig,
)
from Transformer.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(self):
        """
        This method initializes the configuration manager by
        loading the configuration and params yamls and creating
        the artifacts root directory.

        :return: None
        """
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        This method returns the data ingestion configuration after
        creating the required directories.

        :return: DataIngestionConfig entity
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir, config.data_path])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
        )

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
        This method returns the data preprocessing configuration after
        creating the required directories.

        :return: DataPreprocessingConfig entity
        """
        config = self.config.data_preprocessing
        create_directories(
            [config.root_dir, config.preprocessed_data_path, config.tokenizer_path]
        )
        return DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            preprocessed_data_path=Path(config.preprocessed_data_path),
            vocab_size=self.params.vocab_size,
            tokenizer_path=Path(config.tokenizer_path),
        )

    def get_build_model_config(self) -> BuildModelConfig:
        """
        This method returns the build model configuration after
        creating the required directories.

        :return: BuildModelConfig entity
        """
        config = self.config.build_model
        create_directories([config.root_dir])
        return BuildModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            num_layers=self.params.num_layers,
            d_model=self.params.d_model,
            num_heads=self.params.num_heads,
            dff=self.params.dff,
            vocab_size=self.params.vocab_size,
            dropout=self.params.dropout,
            max_length=self.params.max_length,
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        """
        This method returns the model training configuration after
        creating the required directories.

        :return: ModelTrainingConfig entity
        """
        config = self.config.model_training
        create_directories([config.root_dir])
        return ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_path=Path(config.model_path),
            trained_model_path=Path(config.trained_model_path),
            wandb_project_name=config.wandb_project_name,
            wandb_run_name=config.wandb_run_name,
            warmup_steps=config.warmup_steps,
            lr_factor=config.lr_factor,
            clip_grad=config.clip_grad,
            src_tokens_per_batch=config.src_tokens_per_batch,
            tgt_tokens_per_batch=config.tgt_tokens_per_batch,
            total_steps=config.total_steps,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            adam_epsilon=config.adam_epsilon,
            label_smoothing=config.label_smoothing,
            last_n_checkpoints_to_avg=config.last_n_checkpoints_to_avg,
            checkpoint_interval_minutes=config.checkpoint_interval_minutes,
            beam_size=config.beam_size,
            length_penalty=config.length_penalty,
        )
