from Transformer import logger
from Transformer.components.model_training import ModelTraining
from Transformer.config.configuration import ConfigurationManager

STAGE_NAME = "MODEL TRAINING STAGE"


class ModelTrainingPipeline:
    def __init__(self):
        """
        Constructor for ModelTrainingPipeline class.

        Instantiate the configuration manager,
        get the model training configuration and
        create an instance of the ModelTraining component.
        """
        self.config = ConfigurationManager()
        self.model_training_config = self.config.get_model_training_config()
        self.model_training = ModelTraining(config=self.model_training_config)

    def run(self):
        """
        This method starts the model training pipeline.

        :return: None
        """
        self.model_training.train()
        self.model_training.save_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        pipeline = ModelTrainingPipeline()
        pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
