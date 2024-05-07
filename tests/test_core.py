import matplotlib.pyplot as plt
import numpy as np

from photograpy import ImageLayer, FilterGroup

from photograpy import (InvertFilter,
                        ReshapeFilter,
                        FftFilter,
                        IfftFilter)

if __name__ == '__main__':

    filters = [InvertFilter, FftFilter, IfftFilter, InvertFilter]

    fig, ax = plt.subplots(len(filters) + 1, 2)

    arr1 = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            arr1[i, j] = (i + j) / 20 * 255
    im1 = ImageLayer(arr1)
    ax[0, 0].imshow(im1._content)

    arr2 = plt.imread('examples\\image.jpg')
    im2 = ImageLayer(arr2)
    ax[0, 1].imshow(im2._content)

    fg1 = FilterGroup()
    fg2 = FilterGroup()
    im1.add_filter(fg1)
    im2.add_filter(fg2)

    for i in range(len(filters)):
        fg1.append_filter(filters[i]())
        fg2.append_filter(filters[i]())
        ax[i+1, 0].imshow(fg1._filters[-1]._content)
        ax[i+1, 1].imshow(fg2._filters[-1]._content)

    plt.show()
