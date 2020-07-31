import preproc.get_data as data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

from models import unet
from submission import model_to_submission as submission

from models import cnn





def main():

    model = cnn.get_model(None, None, 3, do_compile=False)
    model.load_weights('checkpoints/ckp_cnn_dice_SPECIALDATA.h5')
    submission.create(model, 'ckp_cnn_dice_SPECIALDATA')



    # from models.sdf_model import get_baseline_SDFt, get_flat_tanh_SDFt

    # model = get_baseline_SDFt('mse')
    # model.load_weights('checkpoints/ckp_SDF-tanh_Baseline_with_Unet2_crossval-k0.h5')
    # submission.create(model, "sdf_tanh_unet2_k0")    

    # model = get_flat_tanh_SDFt()
    # model.load_weights('checkpoints/ckp_SDF-tanh_with_scaled_tanh_(0.1),_unet2_crossval-k0.h5')
    # submission.create(model, "sdf_flat_tanh_unet2_k0")    


    # for e in [30,37, 39]:
    #     model = cnn.get_model(None, None, 3, do_compile=False)
    #     model.load_weights('checkpoints/ckp_cnn_dice_EXTDATApp_augmentation_small_e' + str(e) + '.h5')
    #     submission.create(model, 'cnn_dice_EXTDATApp_augmentation_small_e' + str(e))



if __name__ == "__main__":
    main()
