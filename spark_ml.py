import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import RegressionEvaluator
import time

# Step 1: Create a SparkSession
spark = SparkSession.builder.appName("LinearRegression").getOrCreate()

# Step 2: Generate random sample data for linear regression
X = np.random.rand(10000, 11)  # 10000 samples with 11 features
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(10000)  # Linear relationship with random noise

# Step 3: Create a DataFrame from the random data
data = spark.createDataFrame(zip(X.tolist(), y.tolist()), ["features", "label"])

# Step 4: Transform the array column into individual columns
for i in range(11):
    data = data.withColumn(f"feature_{i}", data.features[i])
data = data.drop("features")

# Step 5: Create a VectorAssembler for the transformed features
assembler = VectorAssembler(inputCols=[f"feature_{i}" for i in range(11)], outputCol="features_vector")
data = assembler.transform(data).select("features_vector", "label")

# Step 6: Split the data into training and testing sets
train_data, test_data = data.randomSplit([0.7, 0.3])

# Step 7: Create a LinearRegression object and set its parameters
lr = LinearRegression(featuresCol="features_vector", labelCol="label", maxIter=10, regParam=0.0)

# Step 8: Train the linear regression model on the training data for 100 iterations
start_time = time.time()
for _ in range(100):
    model = lr.fit(train_data)
end_time = time.time()

# Step 9: Make predictions on the test data using the trained model
predictions = model.transform(test_data)

# Step 10: Evaluate the model's performance
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE):", rmse)

# Step 11: Access the model's coefficients and intercept
coefficients = model.coefficients
intercept = model.intercept
print(f"Multiprocessing Time for 100 iterations: {(end_time - start_time) / 60:.3f}min {(end_time - start_time) % 60:.3f}sec")
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Step 12: Stop the SparkSession
spark.stop()
