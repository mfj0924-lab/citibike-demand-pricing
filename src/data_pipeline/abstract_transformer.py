from abc import abstractmethod, ABC
from pyspark.sql import SparkSession


class AbstractTransformer(ABC):
    def __init__(self, spark: SparkSession, config):
        self.spark = spark
        self.config = config

    @abstractmethod
    def transform(self) -> None:
        raise NotImplementedError("Has not been implemented")
