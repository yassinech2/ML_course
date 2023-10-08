import implementations
import numpy as np
import matplotlib.pyplot as plt
import helpers

def load_train_data():
    yb, input_data, ids = helpers.load_csv_data("data/dataset/x_train.csv", sub_sample=False) 
    return yb, input_data, ids

def load_test_data():
    yb, input_data, ids = helpers.load_csv_data("data/dataset/x_test.csv", sub_sample=False) 
    return yb, input_data, ids


def main():
    yb, input_data, ids = load_train_data()
    initial_w = np.zeros(input_data.shape[1])
    max_iters = 1000
    gamma = 0.01
    w, loss = implementations.least_squares(yb, input_data)
    print("Least squares: loss = {}, w = {}".format(loss, w))
    y_pred = np.dot(input_data, w)
    helpers.create_csv_submission(ids, y_pred, "y_test_pred.csv")

if __name__ == '__main__':
    main()
