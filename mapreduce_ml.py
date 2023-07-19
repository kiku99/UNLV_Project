import numpy as np
import pandas as pd

import multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

import time

from tqdm import tqdm


def split_data_into_partitions(X, y, num_partitions):
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
    gradients = np.dot(X.T, np.dot(X, params) - y)

    return gradients


def reduce_function(intermediate_results, learning_rate):
    # Combine gradients and update model parameters
    total_gradients = np.sum(intermediate_results, axis=0)
    updated_params = learning_rate * total_gradients

    return updated_params


def main():
    df = pd.read_csv("train.csv")
    df.drop(["Id"], axis=1, inplace=True)

    df.columns = map(str.lower, df.columns)
    df.rename(columns={"married/single": "married_single"}, inplace=True)

    # --------------------
    # Category cols to num
    cate_cols = ["married_single", "profession", "house_ownership", "car_ownership", "city", "state"]

    for col in cate_cols:
        le = LabelEncoder()
        le = le.fit(df[col])
        df[col] = le.transform(df[col])

    print("[1] Label Encoding-Done.")

    # --------------------
    # Down sampling
    subset_0 = df[df["risk_flag"] == 0]
    subset_1 = df[df["risk_flag"] == 1]

    subset_0_downsampled = resample(subset_0,
                                    replace=False,
                                    n_samples=len(subset_1),
                                    random_state=42)

    df = pd.concat([subset_0_downsampled, subset_1])

    print("[2] Down Sampling-Done.")

    X = df.drop(["risk_flag"], axis=1)
    y = df["risk_flag"].apply(lambda x: int(x))

    # --------------------
    # Data split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val,
                                                      random_state=42)

    # --------------------
    # MinMaxScaler
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values
    print("[3] MinMax Scaling-Done.\n")

    # --------------------
    # Split the data into partitions
    num_partitions = 4
    data_partitions = split_data_into_partitions(X_train_scaled, y_train, num_partitions=num_partitions)

    # Set initial model parameters
    params = np.zeros(X_train_scaled.shape[1])

    # Set learning rate and number of iterations
    learning_rate = 0.1
    num_iterations = 10

    start_time = time.time()
    for _ in tqdm(range(num_iterations)):
        # Create a multiprocessing Pool
        pool = multiprocessing.Pool()
        print(f"Number of worker processes: {pool._processes}")

        # Map step: compute gradients on each data partition in parallel
        intermediate_results = pool.starmap(map_function,
                                            [(data_partition, params) for data_partition in data_partitions])

        # Reduce step: combine gradients and update model parameters
        params = reduce_function(intermediate_results, learning_rate)

        # Close the multiprocessing Pool
        pool.close()
        pool.join()
    end_time = time.time()
    print(f"Multiprocessing Time: {(end_time - start_time) / 60:.3f}min {(end_time - start_time) % 60:.3f}sec")

    # --------------------
    # Prediction
    y_val_pred = np.dot(X_val_scaled, params)
    y_test_pred = np.dot(X_test_scaled, params)

    y_val_pred_binary = np.where(y_val_pred > 0.5, 1, 0)
    y_test_pred_binary = np.where(y_test_pred > 0.5, 1, 0)

    print("Validation Predictions:", y_val_pred_binary)
    print("Test Predictions:", y_test_pred_binary)

    val_acc = accuracy_score(y_val_pred_binary, y_val)
    test_acc = accuracy_score(y_test_pred_binary, y_test)

    print(f"Validation ACC = {val_acc:.3f}")
    print(f"Test ACC = {test_acc:.3f}")


if __name__ == "__main__":
    main()