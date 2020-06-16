import networkx as nx
import numpy as np

def build_graph(pred):
    n = pred.shape[0]
    m = pred.shape[1]
    g = nx.Graph()
    g.add_nodes_from([((i, j), {'class': 0}) for i in range(n) for j in range(m)])
    for i in range(n):
        for j in range(m):
            if i > 0 and pred[i - 1, j] * pred[i, j] > 0.5:
                g.add_edge((i - 1, j), (i, j))
            if i < n - 1 and pred[i + 1, j] * pred[i, j] > 0.5:
                g.add_edge((i + 1, j), (i, j))
            if j > 0 and pred[i, j - 1] * pred[i, j] > 0.5:
                g.add_edge((i, j - 1), (i, j))
            if j < m - 1 and pred[i, j + 1] * pred[i, j] > 0.5:
                g.add_edge((i, j + 1), (i, j))
    return g


def build_graphs(predictions):
    return [build_graph(img) for img in predictions]


def remove_components(g, keep=1):
    """
    removes all but keep (defaults to 1) connected components of g.
    Note: It doesn't actually change the edges, but rather makes the classes of the remaining nodes = 1.

    :param g:
    :param keep: how many components to keep
    :return:
    """
    cc = nx.connected_components(g)

    largest_comps = sorted(cc, key=len, reverse=True)[0:keep]

    for comp in largest_comps:
        for node in comp:
            g.nodes[node]['class'] = 1

    return g


def graph_to_np(g, h, w):
    """
    Turns the graph g back into a numpy array of size (h,w).
    :param g:
    :param h:
    :param w:
    :return:
    """
    assert len(g) == h * w
    a = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            a[i, j] = g.nodes[(i, j)]['class']
    return a
