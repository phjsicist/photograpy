import matplotlib.pyplot as plt
import numpy as np

from photograpy import ImageLayer

from photograpy import InvertFilter

if __name__ == '__main__':

    filter_class = InvertFilter
    args = ()
    kwargs = {}

    fig, ax = plt.subplots(2, 2)

    arr1 = np.empty((10, 10))
    for i in range(10):
        for j in range(10):
            arr1[i, j] = (i + j) / 20 * 255

    im1 = ImageLayer(arr1)
    filter_instance1 = filter_class(*args, **kwargs)
    filter_instance1.apply(im1)
    ax[0, 0].imshow(im1._content)
    ax[1, 0].imshow(filter_instance1._content)

    arr2 = plt.imread('examples\\image.jpg')
    im2 = ImageLayer(arr2)
    filter_instance2 = filter_class(*args, **kwargs)
    filter_instance2.apply(im2)
    ax[0, 1].imshow(im2._content)
    ax[1, 1].imshow(filter_instance2._content)

    plt.show()
