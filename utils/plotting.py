import matplotlib.pyplot as plt


def plot_image_and_mask(image, mask, figsize=(10, 5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax2.imshow(mask, cmap='gray')

    # fig.show()


def plot_mask(mask, figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mask, cmap='gray')
