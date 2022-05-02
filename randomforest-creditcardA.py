#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd 
import numpy 
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


df = pd.read_csv("/Users/Michael1/Desktop/fraud-data1.csv")
dfs = sqlContext.createDataFrame(df)

vectorA = VectorAssembler(inputCols=['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20', 'V21','V22','V23','V24','V25','V26','V27','V28','Amount'], outputCol = "features")

vectorA_df = vectorA.transform(dfs)

label = StringIndexer(inputCol = 'Class', outputCol = 'Fraud')
vectorA_df = label.fit(vectorA_df).transform(vectorA_df)
train, test = vectorA_df.randomSplit([0.7, 0.3], seed = 2018)

rfModel = RandomForestClassifier(featuresCol = 'features', labelCol = "Fraud")
RandomFore = rfModel.fit(train)
predictions = RandomFore.transform(test)

predictions.select("Fraud", "prediction").show(10)

multi = MulticlassClassificationEvaluator(labelCol = 'Fraud', metricName = 'accuracy')
acc = multi.evaluate(predictions)

print("Accuracy is: ", acc*100)


# In[ ]:




