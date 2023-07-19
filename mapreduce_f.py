import numpy as np


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

def map_function(data_partition):
    # Initialize model parameters inside the map function
    params = np.zeros(data_partition[0].shape[1])
    
    # Compute gradients on a data partition using current model parameters
    X, y = data_partition
    gradients = np.dot(X.T, np.dot(X, params) - y)
    return gradients

def reduce_function(intermediate_results, learning_rate):
    # Combine gradients and update model parameters
    total_gradients = np.sum(intermediate_results, axis=0)
    updated_params = learning_rate * total_gradients
    return updated_params

