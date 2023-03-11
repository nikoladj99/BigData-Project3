from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.feature import VectorAssembler
import json

spark = SparkSession.builder.appName("PySpark_Kafka_Consumer").getOrCreate()

df = spark \
  .read \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "kafka:9092") \
  .option("subscribe", "topic2") \
  .option("startingOffsets", "earliest") \
  .option("maxOffsetsPerTrigger", 1) \
  .load()

parsed_df =  df.selectExpr("CAST(value AS STRING)")

split_col = split(parsed_df['value'], ',')
parsed_df = parsed_df.withColumn("user", split_col.getItem(0))
parsed_df = parsed_df.withColumn("check_in_time", split_col.getItem(1))
parsed_df = parsed_df.withColumn("latitude", split_col.getItem(2))
parsed_df = parsed_df.withColumn("longitude", split_col.getItem(3))
parsed_df = parsed_df.withColumn("location_id", split_col.getItem(4))
parsed_df = parsed_df.withColumn("time_spent", split_col.getItem(5))
parsed_df = parsed_df.withColumn("time_stamp", split_col.getItem(6))

parsed_df = parsed_df.where(parsed_df.user != 'user')
parsed_df = parsed_df.withColumn("check_in_time", to_timestamp(parsed_df["check_in_time"], "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"))
parsed_df = parsed_df.withColumn('check_in_time', unix_timestamp('check_in_time'))

parsed_df = parsed_df.withColumn("user", parsed_df["user"].cast(IntegerType()))
parsed_df = parsed_df.withColumn("latitude", parsed_df["latitude"].cast(DoubleType()))
parsed_df = parsed_df.withColumn("longitude", parsed_df["longitude"].cast(DoubleType()))
parsed_df.show()

# Učitajte model za klasifikaciju
model_path = "DecisionTreeModel"
dt_model = DecisionTreeClassificationModel.load(model_path)

# Definišite skup kolona koji će se koristiti za predviđanje
assembler = VectorAssembler(
    inputCols=["user", "check_in_time", "latitude", "longitude"],
    outputCol="features")

# Pretvorite podatke u format koji može da se koristi za predviđanje
parsed_df = assembler.transform(parsed_df)

# Primijenite model za klasifikaciju i dodajte predikcije kao novu kolonu u DataFrame
predictions = dt_model.transform(parsed_df)

predictions = predictions.withColumn("check_in_time", from_unixtime("check_in_time", "yyyy-MM-dd HH:mm:ss.SSS"))
predictions.select("user", "check_in_time", "latitude", "longitude", "prediction") \
           .withColumnRenamed("prediction", "visit_in_worktime") \
           .show(100, False)

filtered_predictions = predictions.filter(predictions.prediction == 1)
filtered_predictions.select("user", "check_in_time", "latitude", "longitude", "prediction") \
           .withColumnRenamed("prediction", "visit_in_worktime") \
           .show()

