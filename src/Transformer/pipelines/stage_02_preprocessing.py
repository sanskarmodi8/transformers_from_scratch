from Transformer import logger
from Transformer.components.preprocessing import Preprocessing
from Transformer.config.configuration import ConfigurationManager

STAGE_NAME = "DATA_PREPROCESSING_STAGE"


class DataPreprocessingPipeline:
    def __init__(self):
        """
        Constructor for DataPreprocessingPipeline class.

        Instantiate the configuration manager,
        get the data preprocessing configuration and
        create an instance of the Preprocessing component.
        """
        self.config = ConfigurationManager()
        self.data_preprocessing_config = self.config.get_data_preprocessing_config()
        self.data_preprocessing = Preprocessing(config=self.data_preprocessing_config)

    def run(self):
        """
        This method starts the data preprocessing pipeline.

        It makes use of the Preprocessing instance to preprocess the specified dataset
        using the configuration details and saves each data split (train, test, validation)
        as a CSV file.

        :return: None
        """
        self.data_preprocessing.train_tokenizer()
        self.data_preprocessing.preprocess_data()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        obj.run()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
