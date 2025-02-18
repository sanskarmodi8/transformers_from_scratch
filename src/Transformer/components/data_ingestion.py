import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from Transformer import logger
from Transformer.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Constructor for DataIngestion class.

        :param config: DataIngestionConfig object
        :return: None
        """
        self.config = config

    def download_dataset(self):
        """
        Downloads the specified dataset using the configuration details and saves
        each data split (train, test, validation) as a CSV file.

        :return: None
        """
        logger.info("Starting dataset download...")

        # Handle dataset download errors
        try:
            dataset = load_dataset("wmt14", "de-en")
            logger.info("Dataset successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

        # Save dataset splits with tqdm progress
        for split in tqdm(dataset.keys(), desc="Saving Splits"):
            df = pd.DataFrame(dataset[split])
            df.to_csv(self.config.data_path / f"{split}.csv", index=False)
            logger.info(
                f"{split} data saved at {self.config.data_path / f'{split}.csv'}"
            )
