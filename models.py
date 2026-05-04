import numpy as np


def least_squares_fit(X, y):
    """
    Fits least squares linear regression from scratch using the Moore-Penrose pseudoinverse.

    This returns the minimum-norm solution, which is important when d > n.
    """
    w_hat = np.linalg.pinv(X) @ y
    return w_hat


def ridge_fit(X, y, alpha=1e-2):
    """
    Fits ridge regression from scratch.

    Objective:
        minimize ||Xw - y||^2 + alpha * ||w||^2

    Closed-form solution:
        w_hat = (X.T @ X + alpha * I)^(-1) @ X.T @ y

    In the code, np.linalg.solve(A, b) is used instead of explicitly computing the inverse,
    where:
        A = X.T @ X + alpha * I
        b = X.T @ y
    """
    d = X.shape[1]

    w_hat = np.linalg.solve(
        X.T @ X + alpha * np.eye(d),
        X.T @ y
    )

    return w_hat

def predict(X, w):
    """
    Makes predictions using a linear model.
    """
    return X @ w


def mean_squared_error(y_true, y_pred):
    """
    Computes mean squared error from scratch.
    """
    errors = y_true - y_pred
    mse = np.mean(errors ** 2)
    return mse


def evaluate_model(X_train, y_train, X_test, y_test, fit_function, **fit_kwargs):
    """
    Fits a model, predicts on train/test sets, and returns train/test MSE.
    """
    w_hat = fit_function(X_train, y_train, **fit_kwargs)

    y_train_pred = predict(X_train, w_hat)
    y_test_pred = predict(X_test, w_hat)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    return train_mse, test_mse, w_hat