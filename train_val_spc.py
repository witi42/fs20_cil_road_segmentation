import datetime
import numpy as np
import sklearn
from sklearn.model_selection import KFold
import preproc.get_data as data
from  metrics.f1 import f1
from metrics.f1 import f1_binary

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


def cross_val(model, model_name, load_training_data=True, x=None, y=None, augment_data_func=None, transform_y=None, use_class_weight=False, epochs=100, batch_size=8, verbose=2):
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
        
        if transform_y != None:
            y_train = transform_y(y_train)
            y_test = transform_y(y_test)
    
        name = model_name + '_crossval-k' + str(index)
        crt_history = fit(model, x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), checkpoint_suffix=name, batch_size=batch_size, verbose=verbose,
                        use_class_weight=use_class_weight)
        histories.append(crt_history)
        
        best_epoch = get_min_index(crt_history.history['loss'])
        best_loss = crt_history.history['loss'][best_epoch]
        best_losses.append(best_loss)
    
        index += 1
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
    #best_model_index = get_min_index(best_losses)
    #model.load_weights("checkpoints/ckp_" + model_name + '_crossval-k' + str(best_model_index) + ".h5")
    #print("best model: checkpoints/ckp_" + model_name + '_crossval-k' + str(best_model_index) + ".h5")




def main():
    from models import simple_patch_conv
    from submission.mask_to_submission import patch_to_label
    num_filters = 1024  # this is chosen somewhat arbitrarily, maybe try some different numbers
    batch_size = 8
    epochs = 200
    def transform_y(y):
        l = []
        num_patches = (int(y.shape[1] / 16), int(y.shape[2] / 16))
        for img_idx in range(y.shape[0]):
            img = np.empty(num_patches)
            for i in range(num_patches[0]):
                for j in range(num_patches[1]):
                    patch = y[img_idx, i*16 : (i+1)*16, j*16 : (j+1)*16]
                    img[i, j] = patch_to_label(patch)
            l.append(img)
        return np.asarray(l)
    
    
    
#################
### Testing Area
###
    
    from visualize.show_img import show_image_single, show_image, show_image_pred, blend_image
    
    x, y = data.get_training_data()
    x, y = data.augment_data(x, y)
    y = transform_y(y)
    x_test, x_test_names = data.get_test_data()
    
    from losses import dice, lovasz
    loss_test = 'binary_crossentropy'
    #loss_test = dice.dice_loss
    #loss_test = lovasz.lovasz_loss
    model = simple_patch_conv.get_model(None, None, 3, num_filters=num_filters, do_compile=False, do_upsampling=False)
    model.compile(optimizer='adam', loss=loss_test, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    model_name = 'spc_test'
    fit(model, x, y, epochs=epochs, validation_split=0.1, validation_data=None, checkpoint_suffix=model_name, batch_size=batch_size, verbose=2)
    
    x_pred = np.squeeze(model.predict(x))
    x_pred_image = (x_pred > 0.5).astype(np.uint8)
    #for i in range(x_pred.shape[0]):
    for i in range(10):
        show_image_pred(x_pred[i], x_pred_image[i], y[i])
    
    x_test_pred = np.squeeze(model.predict(x_test))
    x_test_pred_image = (x_test_pred > 0.5).astype(np.uint8)
    #for i in range(x_test_pred.shape[0]):
    for i in range(10):
        show_image_pred(x_test_pred[i], x_test_pred_image[i], blend_image(x_test[i], np.kron(x_test_pred_image[i], np.ones((16, 16), dtype=int))))
    
###
### Testing Area
#################
    
    
    # spc_cross_entropy
    #model = simple_patch_conv.get_model(None, None, 3, num_filters=num_filters, do_compile=False, do_upsampling=False)
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    #model_name = 'spc_cross_entropy'
    #
    #cross_val(model, model_name, transform_y=transform_y, batch_size=batch_size, epochs=epochs)
    #
    #
    ## spc_balanced_cross_entropy_class_weight
    ##model = simple_patch_conv.get_model(None, None, 3, num_filters=num_filters, do_compile=False, do_upsampling=False)
    ##model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    ##model_name = 'spc_balanced_cross_entropy_class_weight'
    ##
    ##cross_val(model, model_name, transform_y=transform_y, batch_size=batch_size, epochs=epochs)
    #
    #
    ## spc_dice
    #from losses import dice
    #loss = dice.dice_loss
    #model = simple_patch_conv.get_model(None, None, 3, num_filters=num_filters, do_compile=False, do_upsampling=False)
    #model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    #model_name = 'spc_dice'
    #
    #cross_val(model, model_name, transform_y=transform_y, batch_size=batch_size, epochs=epochs)
    #
    #
    ## spc_focal
    #from losses import focal
    #loss = focal.focal_loss
    #model = simple_patch_conv.get_model(None, None, 3, num_filters=num_filters, do_compile=False, do_upsampling=False)
    #model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    #model_name = 'spc_focal'
    #
    #cross_val(model, model_name, transform_y=transform_y, batch_size=batch_size, epochs=epochs)
    
    
    ### spc_lovasz
    ##from losses import lovasz
    ##loss = lovasz.lovasz_loss
    ##model = simple_patch_conv.get_model(None, None, 3, num_filters=num_filters, do_compile=False, do_upsampling=False)
    ##model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    ##model_name = 'spc_lovasz'
    ##
    ##cross_val(model, model_name, transform_y=transform_y, batch_size=batch_size, epochs=epochs)
    
    
    
    
    ## spc_cross_entropy_augmented
    #model = simple_patch_conv.get_model(None, None, 3, num_filters=num_filters, do_compile=False, do_upsampling=False)
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    #model_name = 'spc_cross_entropy_augmented'
    #
    #cross_val(model, model_name, augment_data_func=data.augment_data, transform_y=transform_y, batch_size=batch_size, epochs=epochs)
    #
    #
    ## spc_balanced_cross_entropy_class_weight_augmented
    ##model = simple_patch_conv.get_model(None, None, 3, num_filters=num_filters, do_compile=False, do_upsampling=False)
    ##model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    ##model_name = 'spc_balanced_cross_entropy_class_weight_augmented'
    ##
    ##cross_val(model, model_name, augment_data_func=data.augment_data, transform_y=transform_y, batch_size=batch_size, epochs=epochs)
    #
    #
    ## spc_dice_augmented
    #from losses import dice
    #loss = dice.dice_loss
    #model = simple_patch_conv.get_model(None, None, 3, num_filters=num_filters, do_compile=False, do_upsampling=False)
    #model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    #model_name = 'spc_dice_augmented'
    #
    #cross_val(model, model_name, augment_data_func=data.augment_data, transform_y=transform_y, batch_size=batch_size, epochs=epochs)
    #
    #
    ## spc_focal_augmented
    #from losses import focal
    #loss = focal.focal_loss
    #model = simple_patch_conv.get_model(None, None, 3, num_filters=num_filters, do_compile=False, do_upsampling=False)
    #model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    #model_name = 'spc_focal_augmented'
    #
    #cross_val(model, model_name, augment_data_func=data.augment_data, transform_y=transform_y, batch_size=batch_size, epochs=epochs)
    #
    #
    ## spc_lovasz_augmented
    #from losses import lovasz
    #loss = lovasz.lovasz_loss
    #model = simple_patch_conv.get_model(None, None, 3, num_filters=num_filters, do_compile=False, do_upsampling=False)
    #model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1, f1_binary])
    #model_name = 'spc_lovasz_augmented'
    #
    #cross_val(model, model_name, augment_data_func=data.augment_data, transform_y=transform_y, batch_size=batch_size, epochs=epochs)



if __name__ == "__main__":
    main()
