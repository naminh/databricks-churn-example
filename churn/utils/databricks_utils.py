from pyspark.sql import SparkSession


def get_dbutils():
    try:
        from pyspark.dbutils import DBUtils  # noqa
        if "dbutils" not in locals():
            spark = SparkSession.builder.getOrCreate()
            utils = DBUtils(spark)
            return utils
        else:
            return locals().get("dbutils")
    except ImportError:
        return None
