## 예제 코드
import numpy as np
from pyspark.sql import SparkSession
import time


def split_data_into_partitions(X, y, num_partitions=4):
    # Split the data into partitions
    data_partitions = []
    chunk_size = len(X) // num_partitions

    for i in range(num_partitions):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        X_partition = X[start_idx:end_idx]
        y_partition = y[start_idx:end_idx]
        data_partitions.append((X_partition, y_partition))

    return data_partitions

def map_function(data_partition, params):
    # Compute gradients on a data partition using current model parameters
    X, y = data_partition
    gradients = np.dot(X.T, np.dot(X, params.value) - y)
    return gradients

def reduce_function(intermediate_results, learning_rate):
    # Combine gradients and update model parameters
    total_gradients = np.sum(intermediate_results, axis=0)
    updated_params = learning_rate * total_gradients
    return updated_params

if __name__ == '__main__':
    # Create a SparkSession
    spark = SparkSession.builder.appName("LinearRegression").getOrCreate()

    # Generate sample data for linear regression
    X = np.random.rand(10000, 11)  # 100 samples with 11 features
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(10000)  # Linear relationship with random noise
    print(X.shape)
    print(y.shape)

    # Split the data into partitions
    data_partitions = split_data_into_partitions(X, y)
    # Parallelize the data partitions
    rdd = spark.sparkContext.parallelize(data_partitions)

    # Broadcast the initial model parameters to all workers
    params = spark.sparkContext.broadcast(np.zeros(X.shape[1]))
    print(params.value)

    # Set learning rate and number of iterations
    learning_rate = 0.1
    num_iterations = 100
    
    
    start_time = time.time()

    for _ in range(num_iterations):
        
        # Map step: compute gradients on each data partition in parallel
        intermediate_results = rdd.map(lambda x: map_function(x, params)).collect()

        # Reduce step: combine gradients and update model parameters
        params = spark.sparkContext.broadcast(reduce_function(intermediate_results, learning_rate))
        
    end_time = time.time()
    print(f"Multiprocessing Time: {(end_time - start_time) / 60:.3f}min {(end_time - start_time) % 60:.3f}sec")

    # Use the final parameters for prediction
    X_test = np.random.rand(10, 11)  # New test data
    y_pred = np.dot(X_test, params.value)
    print("Predictions:", y_pred)

    # Stop the SparkSession
    spark.stop()