from Transformer import logger
from Transformer.components.data_ingestion import DataIngestion
from Transformer.config.configuration import ConfigurationManager

STAGE_NAME = "DATA_INGESTION_STAGE"


class DataIngestionPipeline:
    def __init__(self):
        """
        Constructor for DataIngestionPipeline class.

        Instantiate the configuration manager,
        get the data ingestion configuration and
        create an instance of the DataIngestion component.
        """
        self.config = ConfigurationManager()
        self.data_ingestion_config = self.config.get_data_ingestion_config()
        self.data_ingestion = DataIngestion(config=self.data_ingestion_config)

    def run(self):
        """
        This method starts the data ingestion pipeline.

        It makes use of the DataIngestion instance to download the specified dataset
        using the configuration details and saves each data split (train, test, validation)
        as a CSV file.

        :return: None
        """
        self.data_ingestion.download_dataset()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.run()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
