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





def main():
    model = unet.get_model(None, None, 3, do_compile=False)

    model.load_weights('checkpoints/ckp_u_net_focal_EXTDATA_v1.h5')
    sub_name = 'ckp_u_net_focal_EXTDATA_v1.h5'
    submission.create_with_split(model, "cnn_dice_split_EXTDATA")




if __name__ == "__main__":
    main()
