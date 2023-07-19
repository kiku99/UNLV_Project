import numpy as np
import multiprocessing
from mapreduce_f import split_data_into_partitions, map_function, reduce_function
import time

# Generate sample data for linear regression
X = np.random.rand(10000, 11)  # 100 samples with 11 features
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(10000)  # Linear relationship with random noise

if __name__ == '__main__':
    # Split the data into partitions
    data_partitions = split_data_into_partitions(X, y)

    # Set learning rate and number of iterations
    learning_rate = 0.1
    num_iterations = 100

    start_time = time.time()

    for _ in range(num_iterations):
        # Create a multiprocessing Pool
        pool = multiprocessing.Pool()

        # Map step: compute gradients on each data partition in parallel
        intermediate_results = pool.map(map_function, data_partitions)

        # Reduce step: combine gradients and update model parameters
        params = reduce_function(intermediate_results, learning_rate)

        # Close the multiprocessing Pool
        pool.close()
        pool.join()

    end_time = time.time()
    print(f"Multiprocessing Time: {(end_time - start_time) / 60:.3f}min {(end_time - start_time) % 60:.3f}sec")

    # Use the final parameters for prediction
    X_test = np.random.rand(10, 2)  # New test data
    y_pred = np.dot(X_test, params)
    print("Predictions:", y_pred)

