import datetime
import numpy as np
import sklearn
from sklearn.model_selection import KFold
import preproc.get_data as data
from  metrics.f1 import f1
from metrics.f1 import f1_binary
from submission import model_to_submission as submission


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


def cross_val(model, model_name, load_training_data=True, x=None, y=None, augment_data_func=None, use_class_weight=False, epochs=100, batch_size=4, verbose=2):
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
        model.set_weights(reset_weights)  # reset the model weights



    # EVALUATION

    print("\nCROSS-VALIDATION-RESULTS")
    print("model_name: " + model_name)
    print("optimizer: " + str(model.optimizer))
    print("loss: " + str(model.loss))
    print("epochs: 100, early_stopping_patience = 8")


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
    from models.sdf_model import get_CNN_SDFt, get_flat_tanh_CNN_SDFt

    model = get_CNN_SDFt()
    model_name = 'SDF-tanh with Deep Unet ("CNN"), MSE Loss'
    
    cross_val(model, model_name)

    model = get_flat_tanh_CNN_SDFt()
    model_name = 'SDF-tanh with scaled tanh (0.1), Deep Unet ("CNN"), MSE Loss'

    cross_val(model, model_name)



if __name__ == "__main__":
    main()
