import datetime
import numpy as np
from sklearn.model_selection import KFold
import preproc.get_data as data

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# random seed for cross-validation
random_state = 426912378


def fit(model, X_train, Y_train, epochs=100, validation_split=0, validation_data=None, class_weight=None,
        checkpoint_datetime=False, checkpoint_suffix="", batch_size=8):
    earlystopper = EarlyStopping(patience=9, verbose=2)
    suffix = checkpoint_suffix
    if checkpoint_datetime:
        suffix += str(datetime.datetime.now())
    checkpointer = ModelCheckpoint('checkpoints/ckp_{}.h5'.format(suffix), verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=validation_split, validation_data=validation_data,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[earlystopper, checkpointer], class_weight=class_weight,
                        verbose=2)
    return results

def get_min_index(l):
    min_val = float('Inf')
    min_index = -1
    for index in range(len(l)):
        if l[index] < min_val:
            min_val = l[index]
            min_index = index
    return min_index

def cross_val(model, model_name, class_weight=None):
    x, y = data.get_training_data()
    x_norm = x / 255.0

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    histories = []
    index = 0
    for train_index, test_index in kf.split(x_norm):
        x_train, x_test = x_norm[train_index], x_norm[test_index]
        y_train, y_test = y[train_index], y[test_index]

        name = model_name + '_crossval-k' +str(index)
        histories.append(fit(model, x_train, y_train, validation_data=(x_test, y_test), checkpoint_suffix=name,
                             class_weight=class_weight))

        index += 1

    # get used metrics
    keys = histories[0].history.keys()

    # average of metrics over all data splits
    average = {}
    for h in histories:
        best_index = get_min_index(h.history['loss'])
        for k in keys:
            if k not in average:
                average[k] = h.history[k][best_index]
            else:
                average[k] += h.history[k][best_index]
    for k in average:
        average[k] /= len(histories)

    print("\nCross-Validation")
    print("model_name: " + model_name)
    print("optimizer: " + str(model.optimizer))
    print("loss: " + str(model.loss))
    print("epoches: 100, early_stopping_patience = 9")
    print("cross_val_seed: " + random_state)
    print("AVERAGE-METRICS")
    print(average)


def main():
    from models import unet
    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

    model_name = 'u_net_cross_entropy'
    cross_val(model, model_name)


if __name__ == "__main__":
    main()
