# -*- coding: utf-8 -*-
"""Some helper functions for project 1."""
import csv
import numpy as np



def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def build_model_data(yb, input_data):
    """Form (y,tX) to get regression data in matrix form."""
    y = yb
    x = input_data
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def accuracy(y_true, y_pred):
    """Calculate the accuracy of predicted labels."""
    return np.mean(y_true == y_pred)

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def predict(tx,w):
    """predict labels given weights and features in Logoistic Regression"""
    y_pred = sigmoid(np.dot(tx,w))
    y_pred[y_pred <= 0.5] = 0
    y_pred[y_pred > 0.5] = 1
    return y_pred