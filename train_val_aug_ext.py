import datetime
import numpy as np
import sklearn
from sklearn.model_selection import KFold
import preproc.get_data as data
from  metrics.f1 import f1
from metrics.f1 import f1_binary
from submission import model_to_submission as submission
from models import unet

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


def fit(model, X_train, Y_train, epochs=2, validation_split=0, validation_data=None, use_class_weight=False,
        checkpoint_datetime=False, checkpoint_suffix="", batch_size=8, verbose=2):
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(42424242)
    tf.compat.v1.set_random_seed(42424242)

    earlystopper = EarlyStopping(patience=8, verbose=2)
    suffix = checkpoint_suffix
    if checkpoint_datetime:
        suffix += str(datetime.datetime.now())
    os.makedirs("checkpoints", exist_ok = True)
    checkpointer = ModelCheckpoint('checkpoints/ckp_{}.h5'.format(suffix), verbose=2, save_best_only=True)

    class_weight = None
    if use_class_weight:
        flattened_y = Y_train.flatten()
        class_weight = sklearn.utils.class_weight.compute_class_weight('balanced',np.unique(flattened_y),flattened_y)

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


def cross_val(model, model_name, load_training_data=True, x=None, y=None, augment_data_func=None, use_class_weight=False, epochs=100, batch_size=8, verbose=2):
    print('\n\n\n' '5-Cross-Validation: ' + model_name)
    
    if load_training_data:
        x, y = data.get_training_data()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42424242)
    
    reset_weights = model.get_weights()  # for reseting the model weights

    histories = []
    index = 0
    best_losses = []
    for train_index, test_index in kf.split(x):
        print('\nSplit k=' + str(index))
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if augment_data_func != None:
            x_train, y_train = augment_data_func(x_train, y_train)
    
        name = model_name + '_crossval-k' + str(index)
        crt_history = fit(model, x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), checkpoint_suffix=name, batch_size=batch_size, verbose=verbose,
                        use_class_weight=use_class_weight)
        histories.append(crt_history)
        
        best_epoch = get_min_index(crt_history.history['loss'])
        best_loss = crt_history.history['loss'][best_epoch]
        best_losses.append(best_loss)
    
        index += 1

        # create submission for first fold
        if index == 0:
            submission.create(model, name)

        model.set_weights(reset_weights)  # reset the model weights



    # EVALUATION

    print("\nCROSS-VALIDATION-RESULTS")
    print("model_name: " + model_name)
    print("optimizer: " + str(model.optimizer))
    print("loss: " + str(model.loss))
    print("epoches: 100, early_stopping_patience = 8")


    print('\nMETRICS')
    # get used metrics
    keys = histories[0].history.keys()
    # average of metrics over all data splits
    average_metrics = {}
    index = 0
    for h in histories:
        best_index = get_min_index(h.history['loss'])
        current_metrics = {}
        for k in keys:
            if k not in average_metrics:
                average_metrics[k] = h.history[k][best_index]
            else:
                average_metrics[k] += h.history[k][best_index]
            
            current_metrics[k] = h.history[k][best_index]
        print('k='+str(index), current_metrics)
        index += 1
            
    for k in average_metrics:
        average_metrics[k] /= len(histories)


    print("\nAVERAGE-METRICS")
    print(average_metrics)

    # reload best model weights
    #est_model_index = get_min_index(best_losses)
    #model.load_weights("checkpoints/ckp_" + model_name + '_crossval-k' + str(best_model_index) + ".h5")
    #print("best model: checkpoints/ckp_" + model_name + '_crossval-k' + str(best_model_index) + ".h5")





def main():
    # ToDo: - comment out the unnecessary combinations
    #       - set the test-run switch at single_run() below to False
    ext_aug_configs = [                         # desaturated   grayscale   blurred     rotations
                                                #
                       #(None, False, None, 0),  #      -            -          -            0
                                                #
                       (0.5, False, None, 0),   #      X            -          -            0
                       (None, True, None, 0),   #      -            X          -            0
                       (None, False, 1.5, 0),   #      -            -          X            0
                       (None, False, None, 3),  #      -            -          -            3
                                                #
                       (0.5, True, None, 0),    #      X            X          -            0
                       (0.5, False, 1.5, 0),    #      X            -          X            0
                       (0.5, False, None, 3),   #      X            -          -            3
                       (None, True, 1.5, 0),    #      -            X          X            0
                       (None, True, None, 3),   #      -            X          -            3
                       (None, False, 1.5, 3),   #      -            -          X            3
                                                #
                       (0.5, True, 1.5, 0),     #      X            X          X            0
                       (0.5, True, None, 3),    #      X            X          -            3
                       (0.5, False, 1.5, 3),    #      X            -          X            3
                       (None, True, 1.5, 3),    #      -            X          X            3
                                                #
                       (0.5, True, 1.5, 3),     #      X            X          X            3
                       ]
    
    for crt_cfg in ext_aug_configs:
        print("Running with extended data augmentation: " + str(crt_cfg))
        single_run(crt_cfg[0], crt_cfg[1], crt_cfg[2], crt_cfg[3], False)    # set to false for proper execution



def single_run(aug_sat, aug_gs, aug_blur, aug_rr, TEST_RUM_ONLY = False):
    def augment_data_ext(x, y):
        return data.augment_data_extended(x, y, saturation = aug_sat, use_grayscale = aug_gs, blur_amount = aug_blur, num_random_rotations = aug_rr)
    
    epochs = 150
    if TEST_RUM_ONLY:
        epochs = 1
        
    crt_config_name = "_" + str(aug_sat) + "_" + str(aug_gs) + "_" + str(aug_blur) + "_" + str(aug_rr)
    
    # u_net_cross_entropy_augmented
    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    model_name = 'u_net_cross_entropy_augmented_extended' + crt_config_name
    
    cross_val(model, model_name, augment_data_func=augment_data_ext, epochs = epochs)



    # # u_net_balanced_cross_entropy_class_weight_augmented
    # model = unet.get_model(None, None, 3, do_compile=False)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    # model_name = 'u_net_balanced_cross_entropy_class_weight_augmented_extended' + crt_config_name
    #
    # cross_val(model, model_name, augment_data_func=augment_data_ext, epochs = 1)
    #
    #
    #
    # # u_net_dice_augmented
    # from losses import dice
    # loss = dice.dice_loss
    #
    # model = unet.get_model(None, None, 3, do_compile=False)
    # model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    # model_name = 'u_net_dice_augmented_extended' + crt_config_name
    #
    # cross_val(model, model_name, augment_data_func=augment_data_ext, epochs = 1)
    #
    #
    # # u_net_focal_augmented
    # from losses import focal
    # loss = focal.focal_loss
    # model = unet.get_model(None, None, 3, do_compile=False)
    # model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    # model_name = 'u_net_focal_augmented_extended' + crt_config_name
    #
    # cross_val(model, model_name, augment_data_func=augment_data_ext, epochs = 1)
    #
    #
    # # u_net_lovasz_augmented
    # from losses import lovasz
    # loss = lovasz.lovasz_loss
    # model = unet.get_model(None, None, 3, do_compile=False)
    # model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    # model_name = 'u_net_lovasz_augmented_extended' + crt_config_name
    #
    # cross_val(model, model_name, augment_data_func=augment_data_ext, epochs = 1)



if __name__ == "__main__":
    main()
