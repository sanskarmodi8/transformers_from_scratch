import ast
import os

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from Transformer import logger
from Transformer.entity.config_entity import DataPreprocessingConfig


class Preprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """
        Constructor for Preprocessing class.

        :param config: DataPreprocessingConfig object
        :return: None
        """
        self.config = config

    def preprocess_data(self):
        """
        This method preprocesses the data by performing the following operations:
        1. Extracts German and English text from the translation column.
        2. Tokenizes the text using the Helsinki-NLP/opus-mt-de-en tokenizer.
        3. Saves the preprocessed data as a CSV file in the preprocessed_data_path.

        :return: None
        """
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

        for split in tqdm(os.listdir(self.config.data_path), desc="Preprocessing"):
            split_path = os.path.join(self.config.data_path, split)
            df = pd.read_csv(split_path)
            # convert string dict to actual dict
            df["translation"] = df["translation"].apply(lambda x: ast.literal_eval(x))
            # extract eng and german text into diff columns
            df["de"] = df["translation"].apply(lambda x: x["de"])
            df["en"] = df["translation"].apply(lambda x: x["en"])

            df.drop(columns=["translation"], inplace=True)

            # tokenize using bpe tokenizer
            df["de"] = df["de"].apply(
                lambda x: tokenizer.encode(
                    x, truncation=True, padding="max_length", max_length=512
                )
            )
            df["en"] = df["en"].apply(
                lambda x: tokenizer.encode(
                    x, truncation=True, padding="max_length", max_length=512
                )
            )

            df.to_csv(
                os.path.join(self.config.preprocessed_data_path, split), index=False
            )
            logger.info(
                f"Preprocessed {split} data saved at {self.config.preprocessed_data_path / split}.csv"
            )
