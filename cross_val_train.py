import datetime
from sklearn.model_selection import KFold

import preproc.get_data as data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

from submission import mask_to_submission


# random seed for cross-validation
random_state = 426912378


def fit(model, X_train, Y_train, epochs=100, validation_split=0, validation_data=None, class_weight=None,
        checkpoint_datetime=False, checkpoint_suffix="", batch_size=8, verbose=2):
    earlystopper = EarlyStopping(patience=9, verbose=2)
    suffix = checkpoint_suffix
    if checkpoint_datetime:
        suffix += str(datetime.datetime.now())
    os.makedirs("cps", exist_ok = True)
    checkpointer = ModelCheckpoint('cps/ckp_{}.h5'.format(suffix), verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=validation_split, validation_data=validation_data,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[earlystopper, checkpointer], class_weight=class_weight,
                        verbose=verbose)
    return results

def get_min_index(l):
    min_val = float('Inf')
    min_index = -1
    for index in range(len(l)):
        if l[index] < min_val:
            min_val = l[index]
            min_index = index
    return min_index

def cross_val(model, model_name, epochs=100, batch_size=8, verbose=2):

    x, y = data.get_training_data()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    histories = []
    index = 0
    best_losses = []
    current_name = ''
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print('augment data')
        x_train, y_train = data.augment_data_extended(x_train, y_train, num_random_rotations = 3)
    
        current_name = model_name + '_crossval-k' + str(index)
        crt_history = fit(model, x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), checkpoint_suffix=current_name, batch_size=batch_size, verbose=verbose)
        histories.append(crt_history)
        
        best_epoch = get_min_index(crt_history.history['loss'])
        best_loss = crt_history.history['loss'][best_epoch]
        best_losses.append(best_loss)

        model.load_weights(current_name+ '.h5')
    
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
    print("cross_val_seed: " + str(random_state))
    print("AVERAGE-METRICS")
    print(average)

    # reload best model weights
    #best_model_index = get_min_index(best_losses)
    #model.load_weights("cps/ckp_" + model_name + '_crossval-k' + str(best_model_index) + ".h5")
    #print("best model: cps/ckp_" + model_name + '_crossval-k' + str(best_model_index) + ".h5")

    create_sub(model, model_name)

def create_sub(model, sub_name):
    x, x_names = data.get_test_data()

    if not os.path.exists('output'):
        os.makedirs('output')

    for i in range(len(x_names)):
        name = x_names[i][18:]
        print(name)

        pred = model.predict(x[i:i + 1])
        pred = pred.reshape(608, 608)
        pred = (pred > 0.5).astype(np.uint8)

        plt.imsave("output/" + name, pred, cmap=cm.gray)

    if not os.path.exists('submission_csv'):
        os.makedirs('submission_csv')
    submission_filename = 'submission_csv/' + sub_name + '.csv'
    image_filenames = glob.glob('output/*.png')

    mask_to_submission.masks_to_submission(submission_filename, image_filenames)





def main():
    from models import unet

    # focal
    from losses import focal
    loss = focal.focal_loss
    model_name = 'u_net_focal_loss_cross_training_ext_aug_5'
    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss=loss,
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    cross_val(model, model_name)


if __name__ == "__main__":
    main()
