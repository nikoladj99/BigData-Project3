from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.feature import VectorAssembler
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json
import sys

if len(sys.argv) != 2:
    print("Neadekvatan broj argumenata.")
    exit(-1)

spark = SparkSession.builder.appName("Project_3B").getOrCreate()

df = spark \
  .readStream \
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

# Ucitavanje modela
model_path = sys.argv[1]
dt_model = DecisionTreeClassificationModel.load(model_path)

assembler = VectorAssembler(
    inputCols=["user", "check_in_time", "latitude", "longitude"],
    outputCol="features")

parsed_df = assembler.transform(parsed_df)

# Klasifikacija
predictions = dt_model.transform(parsed_df)

predictions = predictions.withColumn("check_in_time", from_unixtime("check_in_time", "yyyy-MM-dd HH:mm:ss.SSS"))
predictions = predictions.withColumn("prediction", when(col("prediction") >= 0.5, 1.0).otherwise(0.0))

# UPIS U INFLUXDB 

def write_to_influxdb(df, epoch_id):
    # Inicijalizacija InfluxDB klijenta
    token = "Xl99hGe7tyPW-wWrgpZRo8vOxA6bK2nR-X3MEoqkigZqnSG1vSqpKOoBmLZdWpWbYKKMKNEfqAAX4FMoKhd5ug=="
    org = "brightkite-org"
    bucket = "brightkite-bucket"
    client = InfluxDBClient(url="http://influxdb:8086", token=token)

    # Kreiranje instance WriteApi klase
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for row in df.collect():
        point = Point("predictions_5b") \
            .field("user", row.user) \
            .field("latitude", row.latitude) \
            .field("longitude", row.longitude) \
            .field("visit_in_worktime", row.prediction)
        write_api.write(bucket=bucket, org=org, record=point)

stream = predictions.writeStream \
    .outputMode("update") \
    .foreachBatch(write_to_influxdb) \
    .trigger(processingTime="10 seconds") \
    .start()

stream.awaitTermination()       






