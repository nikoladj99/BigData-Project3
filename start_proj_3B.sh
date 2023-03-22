#!/bin/bash

spark/bin/spark-submit --master spark://spark-master:7077 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 app_3b.py hdfs://namenode:9000/DecisionTreeModel_5