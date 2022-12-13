from pyspark import SparkConf
from pyspark.sql import SparkSession

BUCKET = "dmacademy-course-assets"
file_before = "vlerick/pre_release.csv"
file_after = "vlerick/after_release.csv"

config = {
    "spark.jars.python": "com.amazonaws.auth.instanceprofilecredentialsprovider"
}