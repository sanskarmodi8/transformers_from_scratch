from Transformer import logger
from Transformer.components.build_model import BuildModel
from Transformer.config.configuration import ConfigurationManager

STAGE_NAME = "MODEL BUILDING STAGE"


class BuildModelPipeline:
    def __init__(self):
        """
        Constructor for BuildModelPipeline class.

        Instantiate the configuration manager,
        get the build model configuration and
        create an instance of the BuildModel component.
        """
        self.config = ConfigurationManager()
        self.build_model_config = self.config.get_build_model_config()
        self.build_model = BuildModel(config=self.build_model_config)

    def run(self):
        """
        This method starts the build model pipeline.

        :return: None
        """
        self.build_model.build()
        self.build_model.save_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        pipeline = BuildModelPipeline()
        pipeline.run()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
