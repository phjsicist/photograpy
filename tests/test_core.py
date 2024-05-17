import matplotlib.pyplot as plt
import numpy as np

from photograpy import LayerGroup
from photograpy.filters import *
from photograpy.masks import *

if __name__ == '__main__':

    filters1 = [SharpenFilter(2, 1), ReshapeFilter((100, 100))]
    filters2 = [SharpenFilter(20, 2), ReshapeFilter((378, 605))]

    arr1 = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            arr1[i, j] = (i + j) / 20 * 255
    im1 = ImageLayer(arr1)

    ft11 = FftFilter()
    im1.add_layer(ft11)
    fmask1 = CustomMask(np.full(ft11.shape, 1))
    ft11.add_mask(fmask1)

    ft12 = IfftFilter(cast_method='clip')
    ft11.add_layer(ft12)

    arr2 = plt.imread('examples\\image2.jpg')
    im2 = ImageLayer(arr2)

    ft21 = FftFilter()
    im2.add_layer(ft21)
    fmask2 = CustomMask(np.full(ft21.shape, 1))
    ft21.add_mask(fmask2)

    ft22 = IfftFilter(cast_method='clip')
    ft21.add_layer(ft22)

    lg1 = LayerGroup()
    lg2 = LayerGroup()

    for i in range(len(filters1)):
        lg1.append_layer(filters1[i])
        lg2.append_layer(filters2[i])

    ft12.add_layer(lg1)
    ft22.add_layer(lg2)

    im1.update()
    im2.update()

    fig, ax = plt.subplots(len(filters1)+3, 2)
    ax[0, 0].imshow(im1.content)
    ax[0, 1].imshow(im2.content)
    ax[1, 0].imshow(ft11.content)
    ax[1, 1].imshow(ft21.content)
    ax[2, 0].imshow(ft12.content)
    ax[2, 1].imshow(ft22.content)

    for i in range(len(filters1)):
        ax[i+3, 0].imshow(lg1.layers[i].content)
        ax[i+3, 1].imshow(lg2.layers[i].content)

    fig, ax = plt.subplots(2, 1)

    ax[0].imshow(ft22.content)
    ax[1].imshow(lg2.content)

    print(im2.update_runtime)
    print(ft21.update_runtime)
    print(ft22.update_runtime)
    print(lg2.update_runtime)

    plt.show()
