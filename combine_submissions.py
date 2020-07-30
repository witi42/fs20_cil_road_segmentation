#!/usr/bin/python

import math
import matplotlib.image as mpimg
import numpy as np
import glob
import postproc.save_submission as subm

files = glob.glob('submissions_to_combine/*.csv')

h = 16
w = h
imgwidth = int(math.ceil((600.0/w))*w)
imgheight = int(math.ceil((600.0/h))*h)
nc = 3

preds = {}

for file in files:
    print("\r " + file, end='')
    f = open(file)
    lines = f.readlines()
    for i in range(1, len(lines)):
        line = lines[i]

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])

        tokens = id.split('_')
        id = tokens[0]
        i = int(tokens[1])
        j = int(tokens[2])

        if id not in preds:
            preds[id] = np.zeros((imgwidth, imgheight), dtype=np.float32)


        je = min(j + w, imgwidth)
        ie = min(i + h, imgheight)
        if prediction == 0:
            adata = np.zeros((w, h))
        else:
            adata = np.ones((w, h))

        preds[id][j:je, i:ie] += adata
    f.close()

print('\nread all files')

for key in preds:
    print('\r' + key, end='')
    preds[key] = preds[key] / float(len(files))

print('did averaging')

preds_np = np.asarray(list(preds.values()))
# print(preds_np)
preds_names = list(map(lambda k: f"input/test_images/test_{int(k)}.png", preds.keys()))

print('saving new preds')
print(preds_names)
subm.save_predictions(preds_np, preds_names, 'ensemble-test', allow_overwrite=True)
