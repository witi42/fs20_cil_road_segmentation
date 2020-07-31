import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (unary_from_softmax)
from skimage import img_as_ubyte
import numpy as np


def crf(original, pred, steps=3, gauss_sxy=2, pairwise_sxy=40, rgb_sxy=11, pairwise_compat=20):
    annotated = np.asarray([1 - pred, pred])

    original = img_as_ubyte(original)
    # annotated = np.moveaxis(annotated, -1, 0)
    annotated = annotated.copy(order='C')

    d = dcrf.DenseCRF2D(original.shape[1], original.shape[0], 2)
    U = unary_from_softmax(annotated)
    # print(U.shape)
    d.setUnaryEnergy(U)

    if isinstance(pairwise_compat, list):
        pairwise_compat = np.asarray(pairwise_compat, dtype=np.float32)

    d.addPairwiseGaussian(sxy=gauss_sxy, compat=20, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=pairwise_sxy, srgb=rgb_sxy, rgbim=original, compat=pairwise_compat,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(steps)
    MAP = np.argmax(Q, axis=0).reshape(original.shape[0], original.shape[1])

    return MAP

