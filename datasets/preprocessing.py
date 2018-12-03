'''
Based on the pre processing method in paper:
"Deep learning of Human Visual Sensitivity in Image Quality Assessment Framework"
Include:
error map generation
low frequency substraction
'''

from skimage.color import rgb2gray
import numpy as np
import scipy.misc as m


def error_map(img1, img2, epsilon=1.):
    assert img1.shape == img2.shape, 'Two inputs should have the same shape!'
    assert len(img1.shape) == 2, 'Inputs should be the grayscale.'

    # Higher value means lower distance
    # range: [0, 1]
    return np.log(1/(((img1-img2)**2)+epsilon/(255**2)))/np.log((255**2)/epsilon)


def low_frequency_sub(img, scale=4):
    assert len(img.shape) == 2, 'Inputs should be the grayscale.'

    img_resize = m.imresize(img, (img.shape[0]/scale, img.shape[1]/scale),
                            interp='nearest')
    img_resize = m.imresize(img_resize, (img.shape[0], img.shape[1]),
                            interp='nearest')

    return (img-img_resize)/255.


if __name__=='__main__':
    img = m.imread('../../data/TID2013_dataset/distorted_images/i01_07_5.bmp')

    img_1 = low_frequency_sub(255 * rgb2gray(img))

    img0 = m.imread('../../data/TID2013_dataset/reference_images/I01.BMP')

    img_0 = low_frequency_sub(255 * rgb2gray(img0))

    error = error_map(img_1, img_0)

    import matplotlib.pyplot as plt
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.subplot(232)
    plt.imshow(img_1, cmap='gray')
    plt.subplot(234)
    plt.imshow(img0, cmap='gray')
    plt.subplot(235)
    plt.imshow(img_0, cmap='gray')
    plt.subplot(233)
    plt.imshow(error, cmap='gray')
    plt.show()

