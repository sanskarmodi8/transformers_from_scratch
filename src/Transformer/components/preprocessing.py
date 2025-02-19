import ast
import os

import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

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

    def train_tokenizer(self):
        """
        Trains a new Byte Pair Encoding (BPE) tokenizer with a shared vocabulary of 37,000 tokens.
        """
        tokenizer = ByteLevelBPETokenizer()

        all_text = []
        for split in os.listdir(self.config.data_path):
            split_path = os.path.join(self.config.data_path, split)
            df = pd.read_csv(split_path)
            df["translation"] = df["translation"].apply(lambda x: ast.literal_eval(x))
            all_text.extend(df["translation"].apply(lambda x: x["de"]).tolist())
            all_text.extend(df["translation"].apply(lambda x: x["en"]).tolist())

        # Save text to a temporary file for training
        temp_text_file = os.path.join(
            self.config.root_dir, "tokenizer_training_text.txt"
        )
        with open(temp_text_file, "w", encoding="utf-8") as f:
            f.write("\n".join(all_text))

        # Train BPE tokenizer
        tokenizer.train(
            files=[temp_text_file],
            vocab_size=37000,
            min_frequency=2,
            special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"],
        )

        # Save tokenizer
        tokenizer.save_model(self.config.tokenizer_path)
        logger.info(f"BPE tokenizer trained and saved at {self.config.tokenizer_path}")

    def preprocess_data(self):
        """
        This method preprocesses the data by performing the following operations:
        1. Extracts German and English text from the translation column.
        2. Tokenizes the text using the trained BPE tokenizer.
        3. Saves the preprocessed data as a CSV file in the preprocessed_data_path.

        :return: None
        """
        # Load trained tokenizer
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(self.config.tokenizer_path, "vocab.json"),
            unk_token="<unk>",
            pad_token="<pad>",
            cls_token="<s>",
            sep_token="</s>",
            mask_token="<mask>",
        )

        for split in tqdm(os.listdir(self.config.data_path), desc="Preprocessing"):
            split_path = os.path.join(self.config.data_path, split)
            df = pd.read_csv(split_path)

            # Convert string dictionary to actual dictionary
            df["translation"] = df["translation"].apply(lambda x: ast.literal_eval(x))

            # Extract German and English text
            df["de"] = df["translation"].apply(lambda x: x["de"])
            df["en"] = df["translation"].apply(lambda x: x["en"])
            df = df.drop(columns=["translation"])

            # Apply BPE tokenization and get token IDs
            df["de"] = df["de"].apply(
                lambda x: tokenizer.encode(
                    x,
                    truncation=True,
                    padding="max_length",
                    max_length=self.config.max_length_tokenizer,
                ).ids
            )
            df["en"] = df["en"].apply(
                lambda x: tokenizer.encode(
                    x,
                    truncation=True,
                    padding="max_length",
                    max_length=self.config.max_length_tokenizer,
                ).ids
            )

            # Save preprocessed data
            df.to_csv(
                os.path.join(self.config.preprocessed_data_path, split), index=False
            )
            logger.info(
                f"Preprocessed {split} data saved at {self.config.preprocessed_data_path}/{split}.csv"
            )
