import matplotlib.pyplot as plt
import numpy as np

from photograpy import ImageLayer, LayerGroup, Mask

from photograpy import (InvertFilter,
                        ReshapeFilter,
                        FftFilter,
                        IfftFilter)

if __name__ == '__main__':

    filters = [FftFilter, IfftFilter]

    fig, ax = plt.subplots(len(filters)+3, 2)

    arr1 = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            arr1[i, j] = (i + j) / 20 * 255
    im1 = ImageLayer(arr1)
    ax[0, 0].imshow(im1.content)

    ft1 = InvertFilter()
    im1.add_layer(ft1)

    mask1 = Mask()
    mask1.content = np.full(ft1.content.shape, 0.5)
    ft1.add_mask(mask1)
    ax[1, 0].imshow(ft1.content)

    arr2 = plt.imread('examples\\image.jpg')
    im2 = ImageLayer(arr2)
    ax[0, 1].imshow(im2.content)

    ft2 = InvertFilter()
    im2.add_layer(ft2)

    mask2 = Mask()
    mask2.content = np.full(ft2.content.shape, 0.5)
    ft2.add_mask(mask2)
    ax[1, 1].imshow(ft2.content)

    lg1 = LayerGroup()
    lg2 = LayerGroup()
    im1.add_layer(lg1)
    im2.add_layer(lg2)

    for i in range(len(filters)):
        lg1.append_layer(filters[i]())
        lg2.append_layer(filters[i]())
        ax[i+2, 0].imshow(lg1.layers[-1].content)
        ax[i+2, 1].imshow(lg2.layers[-1].content)

    lg1.add_layer(ReshapeFilter, (100, 100))
    lg2.add_layer(ReshapeFilter, (52, 200))

    ax[i+3, 0].imshow(lg1.child.content)
    ax[i+3, 1].imshow(lg2.child.content)

    plt.show()
