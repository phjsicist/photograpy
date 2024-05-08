import matplotlib.pyplot as plt
import numpy as np

from photograpy import ImageLayer, LayerGroup

from photograpy import (InvertFilter,
                        ReshapeFilter,
                        FftFilter,
                        IfftFilter)

if __name__ == '__main__':

    filters = [InvertFilter, FftFilter, IfftFilter, InvertFilter]

    fig, ax = plt.subplots(len(filters)+2, 2)

    arr1 = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            arr1[i, j] = (i + j) / 20 * 255
    im1 = ImageLayer(arr1)
    ax[0, 0].imshow(im1.content)

    arr2 = plt.imread('examples\\image.jpg')
    im2 = ImageLayer(arr2)
    ax[0, 1].imshow(im2.content)

    lg1 = LayerGroup()
    lg2 = LayerGroup()
    im1.add_layer(lg1)
    im2.add_layer(lg2)

    for i in range(len(filters)):
        lg1.append_layer(filters[i]())
        lg2.append_layer(filters[i]())
        ax[i+1, 0].imshow(lg1.layers[-1].content)
        ax[i+1, 1].imshow(lg2.layers[-1].content)

    lg1.add_layer(ReshapeFilter, (100, 100))
    lg2.add_layer(ReshapeFilter, (52, 200))

    ax[i+2, 0].imshow(lg1.child.content)
    ax[i+2, 1].imshow(lg2.child.content)

    plt.show()
