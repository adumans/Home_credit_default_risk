from __future__ import print_function
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# Read data
spark = SparkSession.builder.appName("pyspark data source").getOrCreate()
df = spark.read.load("../data/training.csv", format="csv", header="true")
test_df = spark.read.load("../data/testing.csv", format="csv", header="true")

# Convert DataType and DataFrame
features_name = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
for name in features_name:
    df = df.withColumn(name, df[name].cast("double"))
    df_test = df_test.withColumn(name, df[name].cast("double"))
df = df.withColumn("indexedLabel", df["SK_ID_CURR"].cast("double"))
df_test = df_test.withColumn("indexedLabel", df["SK_ID_CURR"].cast("double"))

# Assemble features as sparse Vector
assembler = VectorAssembler(inputCols=features_name, outputCol= "features")
features = assembler.transform(df)
df = df.withColumn("indexedFeatures", features)

trainingData = df.select(["indexedLabel", "indexedFeatures"])
testData = df_test.select("indexedFeatures")
del df, df_test

# Train a GBT model.
gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)

# Train model.
model = gbt.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

gbtModel = model.stages[2]
print(gbtModel)  # summary only

spark.stop()