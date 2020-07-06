def get_combined_loss (loss_1, loss_2, ratio = 0.5):
    """
    Returns a function that calculates ratio * loss_1 + (1 - ratio) * loss_2
    :param ratio: should be in [0,1] for it to make sense
    :return: a loss function that is a combination of the two losses provided
    """
    def loss(y_true, y_pred):
        return ratio * loss_1(y_true, y_pred) + (1 - ratio) * loss_2(y_true, y_pred)

    return loss

