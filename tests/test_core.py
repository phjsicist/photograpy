import matplotlib.pyplot as plt
import numpy as np

from photograpy import ImageLayer, LayerGroup, CustomMask

from photograpy import (InvertFilter,
                        ReshapeFilter,
                        FftFilter,
                        IfftFilter)

if __name__ == '__main__':

    filters1 = [InvertFilter(), ReshapeFilter((100, 100))]
    filters2 = [InvertFilter(), ReshapeFilter((100, 100))]

    arr1 = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            arr1[i, j] = (i + j) / 20 * 255
    im1 = ImageLayer(arr1)

    ft11 = FftFilter()
    im1.add_layer(ft11)
    fmask1 = CustomMask(np.full(ft11.shape, 1))
    ft11.add_mask(fmask1)

    ft12 = IfftFilter(cast_method='squeeze')
    ft11.add_layer(ft12)

    ft13 = InvertFilter()
    ft12.add_layer(ft13)

    mask1 = CustomMask(np.full(ft13.shape, 0.75))
    ft13.add_mask(mask1)

    arr2 = plt.imread('examples\\image.jpg')
    im2 = ImageLayer(arr2)

    ft21 = FftFilter()
    im2.add_layer(ft21)
    fmask2 = CustomMask(np.full(ft21.shape, 1))
    ft21.add_mask(fmask2)

    ft22 = IfftFilter(cast_method='squeeze')
    ft21.add_layer(ft22)

    ft23 = InvertFilter()
    ft22.add_layer(ft23)

    mask2 = CustomMask(np.full(ft23.shape, 0.75))
    ft23.add_mask(mask2)

    lg1 = LayerGroup()
    lg2 = LayerGroup()
    ft13.add_layer(lg1)
    ft23.add_layer(lg2)

    for i in range(len(filters1)):
        lg1.append_layer(filters1[i])
        lg2.append_layer(filters2[i])

    im1.update()
    im2.update()

    fig, ax = plt.subplots(len(filters1)+3, 2)
    ax[0, 0].imshow(im1.content)
    ax[0, 1].imshow(im2.content)
    ax[1, 0].imshow(ft12.content)
    ax[1, 1].imshow(ft22.content)
    ax[2, 0].imshow(ft13.content)
    ax[2, 1].imshow(ft23.content)

    for i in range(len(filters1)):
        ax[i+3, 0].imshow(lg1.layers[i].content)
        ax[i+3, 1].imshow(lg2.layers[i].content)

    plt.show()
