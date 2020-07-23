import preproc.get_data as data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob
import os
from submission import mask_to_submission

def create(model, sub_name):
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