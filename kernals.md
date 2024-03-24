# Kernals

- A kernel is essentially a fixed size array of numerical coefficients along with an anchor point in that array, which is typically located at the center. They are masks or filters that modify pixel values based on their surrounding pixels. They are matrices where each element represents the weight given to a corresponding pixel in the neighborhood.
- Convolution with a kernel involves multiplying the kernel with the pixel values in a neighborhood and summing the results to determine the new pixel value.
- Different types of kernels include those for blurring, edge detection, sharpening, and more, each with specific properties and effects on the image.

## Working of Convolution

1. Place the kernel anchor on top of a determined pixel, with the rest of the kernel overlaying the corresponding local pixels in the image.
2. Multiply the kernel coefficients by the corresponding image pixel values and sum the result.
3. Place the result to the location of the anchor in the input image.
4. Repeat the process for all pixels by scanning the kernel over the entire image.

Expressing the procedure above in the form of an equation we would have:
![convolution](https://github.com/ToheedAsghar/Practice_CV/assets/121859513/0097886c-d834-4cde-821d-061d1014ef85)


## Blurring Using Kernals

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolution(img, kernel, average=False):
    kh, kw = kernel.shape
    ih, iw = img.shape

    pad: int = kh // 2
    paddedImage = np.zeros((ih + pad*2, iw + pad*2))
    outputImage = np.zeros(img.shape)

    # dimensions of padded image
    x, y = paddedImage.shape
    paddedImage[pad:x - pad, pad:y - pad] = img

    averageDiv = kw * kh
    for i in range(ih):
        for j in range(iw):
            outputImage[i][j] = np.sum(kernel*paddedImage[i:i+kh, j:j+kw])
            if average:
                outputImage[i][j] /= averageDiv

    return outputImage

img = cv2.imread('Images/Huzaifa.png')
kernel = np.ones((3, 3), np.float32) / 9

blurred_img = convolution(img, kernel)

f, plots = plt.subplots(1, 2)
plots[0].imshow(img, cmap='gray')
plots[1].imshow(blurred_img, cmap='gray')
plt.show()



```

## Explanation

```python
kernel = np.ones((3, 3), np.float32) / 9
```

This creates a 3x3 NumPy array filled with ones of data type np.float32. The shape (3, 3) represents a 3x3 matrix.
Divide operation (i.e /9) is used to create a kernel for blurring that averages the pixel values in a 3x3 neighborhood.

```python
k = np.sum(roi * kernel, axis=(0, 1))
```

**roi * kernel:** This performs element-wise multiplication between the region of interest (roi) and the kernel. Each element in the roi is multiplied by the corresponding element in the kernel.

**np.sum(...):** This calculates the sum of the resulting array obtained from element-wise multiplication. It sums up all the products of elements in the roi and kernel.

**axis=(0, 1):** This specifies along which axes to perform the sum operation. Here, (0, 1) indicates that the sum is calculated along axis 0 (rows) and axis 1 (columns), effectively summing up all elements in each channel of the resulting array.


## Kernals

```python
meanBlur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
blur = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.250, 0.125], [0.0625, 0.125, 0.0625]])
sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
```

## Inbuilt-Kernel Convolution using cv2.filter2D()

```python
import cv2
import numpy as np

image = cv2.imread('Images/Huzaifa.png')
kernel = np.ones((3, 3), np.float32) / 9

blurred_img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

cv2.imshow('Original', image)
cv2.imshow('Blurred Image', blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
#### cv2.filter2D Syntax:

```python
 dst = cv.filter2D(src, ddepth, kernel)
```

The arguments denote:
- src: Source image
- dst: Destination image
- ddepth: The depth of dst. A negative value (such as −1) indicates that the depth is the same as the source.
- kernel: The kernel to be scanned through the image

[cv2.filter2d Blog by OpenCV](https://docs.opencv.org/3.4/d4/dbd/tutorial_filter_2d.html)

**GaussianBlur**
> blur = cv.GaussianBlur(img,(5,5),0)

**Median Blurring**
> median = cv.medianBlur(img,5)

**Bilateral Filtering**
> blur = cv.bilateralFilter(img,9,75,75)

