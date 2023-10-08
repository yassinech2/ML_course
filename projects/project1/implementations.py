import numpy as np

def compute_gradient_mse(y, tx, w):
        """Compute the gradient."""
        e = y - tx.dot(w)
        return - tx.T.dot(e) / len(e)
    
def compute_loss_mse(y, tx, w):
    """Calculate the loss using mse."""
    e = y - tx.dot(w)
    return e.dot(e) / (2 * len(e))


def mean_squared_error_gd(y, tx, initial_w,max_iters, gamma) :
    """Calculate the loss using mse""" 
    w = initial_w
    for iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
        loss = compute_loss_mse(y, tx, w)
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    
    return (w,loss)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def mean_squared_error_sgd(y, tx, initial_w,max_iters, gamma):
    """Calculate the loss using mse"""
    w = initial_w
    for iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
            w = w - gamma * gradient
            loss = compute_loss_mse(y, tx, w)
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    
    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss_mse(y, tx, w)
    return w, loss

def ridge_regression(y,tx,lambda_=0.5):
    """implement ridge regression."""
    N = tx.shape[0]
    D = tx.shape[1]
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * np.identity(D), tx.T.dot(y))
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    epsilon = 1e-15  # Small positive value
    pred = np.clip(pred, epsilon, 1 - epsilon)
    
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred - y)
    return gradient / len(y)

def logistic_regression(y, tx, initial_w,max_iters, gamma):
    """implement logistic regression.
    Used Gradient Descent as optimization method"""
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = compute_logistic_loss(y, tx, w)
        gradient = compute_gradient_logistic(y, tx, w)
        w = w - gamma * gradient
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w,max_iters, gamma):
    """implement regularized logistic regression
    Used Gradient Descent as optimization method"""
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = compute_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
    return w, loss