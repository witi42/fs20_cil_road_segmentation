import matplotlib.pyplot as plt


def plot(images, labels=None, title=None):
    n = len(images)

    width = 3
    if n < 3:
        width = n

    # three images per row
    height = (n + 2) // 3
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title, fontsize=16)
    for i in range(n):
        s = fig.add_subplot(height, width, i + 1)

        if labels is not None:
            s.set_title(labels[i])

        plt.imshow(images[i])
