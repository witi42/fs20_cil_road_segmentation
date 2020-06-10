import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """
    predicts the average accuracy over all images. Both labels must have the same shape, namely
    (n_samples, height, width) or possibly (n_samples, height, width, 1)
    :param y_true: The true groundtruth labels
    :param y_pred: The predicted labes
    :return: the average accuracy
    """
    if y_true.ndim == 4:
        y_true = y_true[:, :, :, 0]
    if y_pred.ndim == 4:
        y_pred = y_pred[:, :, :, 0]

    if not y_true.shape == y_pred.shape:
        raise ValueError("Must have same shapes!")

    if len(np.unique(y_pred)) > 2:
        raise ValueError("y_pred should only be zeros and ones")

    n, h, w = y_true.shape
    agreement = y_true == y_pred

    return np.sum(agreement) / (n*h*w)