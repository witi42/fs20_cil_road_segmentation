def convert_to_sdft_loss(loss_fn):
    """
    Takes a tf. loss function that expects values in [0,1] / {0,1} and returns one that can handle
    GT in (-inf, inf), where <= 0 corresponds to street and the predictions are in [-1,1], where again [-1, 0]
    corresponds to street.

    :param loss_fn:
    :return:
    """

    def _converted_loss(y_true, y_pred):
        return loss_fn(y_true <= 0, - (y_pred + 1) / 2)

    return _converted_loss
