from joblib import Parallel, delayed
import tensorflow as tf
import numpy as np


def closest_cell(mat):
    """
    Takes a binary tensor M of shape (n, m) and for each position (i,j) computes
    min_{k,l s.th. M[k,l] = 1} \\(i,j) - (k,l)\\
    :param mat: binary tensor of shape (n,m)
    :return: a tensor of shape (n,m) as defined above
    """
    n, m = mat.shape
    g = [[] for _ in range(n * m)]

    queue = []
    dist = np.asarray([[np.inf for _ in range(m)] for _ in range(n)])
    visit = [False for _ in range(n * m)]

    # building the graph. each pixel is a node connected to its neighbours.
    for i in range(n):
        for j in range(m):
            if i < n - 1:
                g[i * m + j].append((i + 1) * m + j)
                g[(i + 1) * m + j].append(i * m + j)
            if j < m - 1:
                g[i * m + j].append(i * m + j + 1)
                g[i * m + j + 1].append(i * m + j)

            # add all the positive pixels to the BFS Queue
            if mat[i, j]:
                dist[i, j] = 0
                visit[i * m + j] = True
                queue.append(i * m + j)

    #  Performing BFS starting at the positive pixels
    while queue:
        tmp = queue.pop(0)
        tmp_i = tmp // m
        tmp_j = tmp % m
        # print(tmp)
        for nbr in g[tmp]:
            # print("nbr:", nbr)
            if not visit[nbr]:
                nbr_i = nbr // m
                nbr_j = nbr % m
                dist[nbr_i, nbr_j] = min(dist[nbr_i, nbr_j], dist[tmp_i, tmp_j] + 1)
                queue.append(nbr)
                visit[nbr] = True

    return dist


def sdf(images):
    """
    Takes a binary tensor of shape (batch_size, h, w) and computes the signed distance function
    based on it.

    Def.
    ----
    Let p be a pixel in image and let E denote the edge of the street (positive region of the binary mask.) Then,
                { - d(p, E)     if I[p] = 1, i.e. if the pixel belongs to the street
    sdf(p) :=   {
                { d(p, E)       otherwise,

    where d(p, E) = min_{q \in E} ||p - q||

    :param images: tensor
    :return: a tensor of shape (batch_size, h, w) as defined above.
    """
    images = tf.cast(images, tf.float32)
    edges = tf.image.sobel_edges(images[:, :, :, None])
    edges = tf.sqrt(edges[:, :, :, 0, 0] ** 2 + edges[:, :, :, 0, 1] ** 2)
    edges /= tf.sqrt(32.)

    # edges mask, true iff pixel is an edge
    edges = edges > 0.5

    # cc = Parallel(n_jobs=10)(delayed(closest_cell)(edges[i].numpy()) for i in range(len(edges)))
    cc = [closest_cell(edge.numpy()) for edge in edges]
    cc = tf.constant(cc)

    return tf.where(images == 1., -cc, cc)
