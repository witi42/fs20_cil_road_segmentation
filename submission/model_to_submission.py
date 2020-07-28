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


def create_with_split(model, sub_name):
    small_s = 400
    large_s = 608

    x, x_names = data.get_test_data()

    if not os.path.exists('output'):
        os.makedirs('output')

    print("Predicting Test Images")
    for i in range(len(x_names)):
        name = x_names[i][18:]
        print(name)

        image = x[i:i + 1][0]
        aa = image[0:small_s, 0:small_s]
        ab = image [0:small_s, large_s-small_s: large_s]
        ba = image [large_s-small_s: large_s, 0:small_s]
        bb = image [large_s-small_s: large_s, large_s-small_s: large_s]

        x_splits = np.array([aa, ab, ba, bb])

        pred = model.predict(x_splits)
        pred = pred.reshape(4, small_s, small_s)
        pred = (pred > 0.5).astype(np.uint8)

        aa_ = pred[0, 0:large_s // 2, 0:large_s // 2]
        ab_ = pred[1, 0:large_s // 2, small_s - large_s // 2:small_s]
        ba_ = pred[2, small_s - large_s // 2:small_s, 0:large_s // 2]
        bb_ = pred[3, small_s - large_s // 2:small_s, small_s - large_s // 2:small_s]

        a = np.concatenate((aa_, ab_), axis=1)
        b = np.concatenate((ba_, bb_), axis=1)
        out = np.concatenate((a, b), axis=0)

        plt.imsave("output/" + name, out, cmap=cm.gray)

    if not os.path.exists('submission_csv'):
        os.makedirs('submission_csv')
    submission_filename = 'submission_csv/' + sub_name + '.csv'
    image_filenames = glob.glob('output/*.png')

    mask_to_submission.masks_to_submission(submission_filename, image_filenames)