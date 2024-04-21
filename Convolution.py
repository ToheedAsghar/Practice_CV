import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def convolve(img, kernel):

    kernel = np.flipud(np.fliplr(kernel))
    
    kx, ky = kernel.shape
    ix, iy = img.shape[:2]

    pad = kx // 2

    # img = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT)

    # defining a buffer for padded image
    pImage = np.zeros((ix + pad*2, iy + pad*2))

    # copying the img to the coresponding portion in the padded Image
    pImage[pad:pad+ix, pad:pad+iy] = np.copy(img)

    # defining bufer for output image
    out = np.copy(img)

    for i in range(pad, ix):
        for j in range(pad, iy):
            roi = pImage[i - pad:i+pad+1, j-pad:j+pad+1]
            out[i][j] = (kernel*roi).sum()

    return out


img = cv.imread('huzaifa.png', 0)
kernal = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
img = convolve(img, kernal)

plt.imshow(img, cmap='gray')
plt.show()
