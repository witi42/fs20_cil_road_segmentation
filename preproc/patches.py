import numpy as np

def blockshaped_dataset(data, nrows, ncols):
  """
  Takes a dataset of shape (n_images, h, w, channels), and returns a
  new dataset of shape (n', nrows, ncols, channels), where each element
  is a (nrows x ncols) patch of the old images.

  """
  n_imgs, h, w, c = data.shape
  assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
  assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
  return (data.reshape(n_imgs, h//nrows, nrows, -1, ncols, c)
               .swapaxes(2,3)
               .reshape(-1, nrows, ncols, c))

def unblockshaped_dataset(data, h, w):
  """
  If dataset is a set of patches (n', nrows, ncols, channels) and reconstructs the
  original dataset of shape (n_images, h, w, channels)
  """
  n, nrows, ncols, c = data.shape
  blocks = (h //nrows) * (w//ncols)
  return (data.reshape(n // blocks, h//nrows, -1, nrows, ncols, c)
              .swapaxes(2,3)
              .reshape(-1, h, w, c))
