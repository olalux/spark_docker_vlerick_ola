from pyspark import SparkConf
from pyspark.sql import SparkSession

BUCKET = ""
KEY = ""

config = {
    "spark.jars.python": "com.amazonaws.auth.instanceprofilecredentialsprovider"
}