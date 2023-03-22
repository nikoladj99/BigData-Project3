#!/bin/bash

spark/bin/spark-submit --master spark://spark-master:7077 app_3a.py hdfs://namenode:9000/data_3.csv hdfs://namenode:9000/DecisionTreeModel_12
