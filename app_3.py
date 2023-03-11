import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql.functions import unix_timestamp
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics

if len(sys.argv) != 2:
    print("Neadekvatan broj argumenata.")
    exit(-1)

spark = SparkSession.builder.appName("Project_3").getOrCreate()

# Ucitavanje podataka
df = spark.read.option("inferSchema", True).option("header", True).csv(sys.argv[1])

# Kreiranje kolone koja sadrži sat posete
df_hour = df.withColumn('visit_hour', hour('check_in_time'))

# Kreiranje kolone 'visit_in_worktime' koja će imati vrednost 'True' ako je poseta u radno vreme, a 'False' ako nije
df_worktime = df_hour.withColumn('visit_in_worktime', when((col('visit_hour') >= 9) & (col('visit_hour') < 17), True).otherwise(False))

df_worktime = df_worktime.drop('visit_hour', 'time_spent', 'location_id')

# df_worktime.show()

# Pretvaranje u UNIX vremensku oznaku
df_worktime = df_worktime.withColumn('check_in_time', unix_timestamp('check_in_time'))


# Podela podataka na skup za treniranje i skup za testiranje
train, test = df_worktime.randomSplit([0.7, 0.3], seed=42)

# Konvertovanje kolone 'visit_in_worktime' u numericki tip
train = train.withColumn('visit_in_worktime', when(col('visit_in_worktime') == True, 1).otherwise(0))
test = test.withColumn('visit_in_worktime', when(col('visit_in_worktime') == True, 1).otherwise(0))

# Kreiranje vektora karakteristika
feature_cols = ["user", "check_in_time", "latitude", "longitude"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Transformacija skupa za treniranje i skup za testiranje
train = assembler.transform(train).select("features", "visit_in_worktime")
test = assembler.transform(test).select("features", "visit_in_worktime")

# Kreiranje i treniranje Decision Tree klasifikatora
dt = DecisionTreeClassifier(featuresCol="features", labelCol="visit_in_worktime")
model = dt.fit(train)
model.save("DecisionTreeModel")

# Predviđanje klasa za skup za testiranje
predictions = model.transform(test)

# Evaluacija performansi klasifikatora

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="visit_in_worktime", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Tacnost modela: ", accuracy)

# Evaluacija F1 score-a
f1_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="visit_in_worktime", metricName="f1")
f1_score = f1_evaluator.evaluate(predictions)
print("F1 skor modela: ", f1_score)

# Evaluacija odziva (recall)
recall_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="visit_in_worktime", metricName="weightedRecall")
recall = recall_evaluator.evaluate(predictions)
print("Odziv modela: ", recall)

# Evaluacija preciznosti (precision)
precision_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="visit_in_worktime", metricName="weightedPrecision")
precision = precision_evaluator.evaluate(predictions)
print("Preciznost modela: ", precision)

print("Raspodela po klasama")
predictions.groupBy("visit_in_worktime").count().show()

# Konvertovanje tipa podatka kolone "visit_in_worktime"
predictions = predictions.withColumn("visit_in_worktime", col("visit_in_worktime").cast("double"))

# Konverzija predikcija i etiketa u RDD
predictionAndLabels = predictions.select("prediction", "visit_in_worktime").rdd

# Kreiranje instance klasifikacione metrike
metrics = MulticlassMetrics(predictionAndLabels)

# Štampanje matrice konfuzije
print("Matrica konfuzije:")
print(metrics.confusionMatrix().toArray())


