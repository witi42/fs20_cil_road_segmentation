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


def plot_training_history(history):
    # From tf tutoial
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = len(loss)

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()