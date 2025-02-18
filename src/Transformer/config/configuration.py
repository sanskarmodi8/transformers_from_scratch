from pathlib import Path

from Transformer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from Transformer.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
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
        create_directories([config.root_dir, config.preprocessed_data_path])
        return DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            preprocessed_data_path=Path(config.preprocessed_data_path),
        )
