import numpy as np


def generate_base_data(n_train=100, n_test=1000, max_d=300, d_true=10, noise_std=0.5, seed=42):
    """
    Generates one synthetic dataset for the whole double descent experiment.

    The target y is generated only from the first d_true features.
    The remaining features are noise/irrelevant features.
    """
    rng = np.random.default_rng(seed)

    X_train_full = rng.normal(size=(n_train, max_d))
    X_test_full = rng.normal(size=(n_test, max_d))

    # generates a vector of zeros
    w_star = np.zeros(max_d)

    # fills it up until d_true with random true weights
    w_star[:d_true] = rng.normal(size=d_true) / np.sqrt(d_true)

    # y = Xw* + ε
    y_train = X_train_full @ w_star + rng.normal(scale=noise_std, size=n_train)
    y_test = X_test_full @ w_star + rng.normal(scale=noise_std, size=n_test)

    return X_train_full, y_train, X_test_full, y_test, w_star