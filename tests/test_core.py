import matplotlib.pyplot as plt
import numpy as np

from photograpy import ImageLayer

from photograpy import InvertFilter, ReshapeFilter

if __name__ == '__main__':

    filter_class1 = ReshapeFilter
    args1 = ((100, 100),)
    kwargs1 = {}

    filter_class2 = InvertFilter
    args2 = ()
    kwargs2 = {}

    fig, ax = plt.subplots(3, 2)

    arr1 = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            arr1[i, j] = (i + j) / 20 * 255

    im1 = ImageLayer(arr1)
    ax[0, 0].imshow(im1._content)
    filter_instance11 = filter_class1(*args1, **kwargs1)
    filter_instance11.apply(im1)
    ax[1, 0].imshow(filter_instance11._content)
    filter_instance12 = filter_class2(*args2, **kwargs2)
    filter_instance12.apply(filter_instance11)
    ax[2, 0].imshow(filter_instance12._content)

    arr2 = plt.imread('examples\\image.jpg')

    im2 = ImageLayer(arr2)
    ax[0, 1].imshow(im2._content)
    filter_instance21 = filter_class1(*args1, **kwargs1)
    filter_instance21.apply(im2)
    ax[1, 1].imshow(filter_instance21._content)
    filter_instance22 = filter_class2(*args2, **kwargs2)
    filter_instance22.apply(filter_instance21)
    ax[2, 1].imshow(filter_instance22._content)

    plt.show()
