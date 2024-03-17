# Sobel Filter

The Sobel operator is used to highlight edges within an image by estimating the first derivative of the image. It involves convolving the image with two special kernels, one for detecting vertical edges and the other for detecting horizontal edges. The result of the Sobel operator at each point in the image is either the corresponding gradient vector or the norm of this vector. This means that it emphasizes the areas in the image where the intensity changes rapidly, which often corresponds to the presence of edges.

```python
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('eye.png', 0)
assert img is not None, "file could not be read, check with os.path.exists()"

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

sx = cv.filter2D(img, -1, sobel_x)
sy = cv.filter2D(img, -1, sobel_y)

plt.subplot(141)
plt.imshow(img, cmap='gray')
plt.subplot(142)
plt.imshow(sx, cmap='gray')
plt.subplot(143)
plt.imshow(sy, cmap='gray')

sobel_combined = cv.addWeighted(sx, 0.5, sy, 0.5, 0)
plt.subplot(144)
plt.imshow(sobel_combined, cmap='gray')
plt.show()

```

![sobel](https://github.com/ToheedAsghar/Practice_CV/assets/121859513/a5bf698e-f65b-4c33-b282-6b887222c280)

## cv2.sobel()

#### Syntax

```python
cv2.Sobel(src, ddepth, dx, dy, ksize)
```
Where:

- src: Input image.
- ddepth: Depth of the output image.
- dx and dy: Specify whether Sobel-x or Sobel-y is to be used.
- ksize: Kernel size.

To use the *Sobel filter* in OpenCV, you can apply the cv2.Sobel() function, specifying the precision of the output image and the order of the derivative in each direction. For example, if dx=1 and dy=0, the function computes the 1st derivative Sobel image in the x-direction. If both dx=1 and dy=1, it computes the 1st derivative Sobel image in both directions.

#### Note

*Black-to-White transition is taken as Positive slope (it has a positive value) while White-to-Black transition is taken as a Negative slope (It has negative value). So when you convert data to np.uint8, all negative slopes are made zero. In simple words, you miss that edge.*

*If you want to detect both edges, better option is to keep the output datatype to some higher forms, like cv.CV_16S, cv.CV_64F etc, take its absolute value and then convert back to cv.CV_8U.*

#### Demonstration

```python
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('eye.png')
assert img is not None, "file could not be read, check with os.path.exists()"

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grad_x = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=3)
grad_y = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=3)

grad = cv.sqrt(grad_x**2 + grad_y**2)
grad_norm = (grad * 255 / grad.max()).astype('uint8')

plt.subplot(221)
plt.imshow(gray_img, cmap='gray')
plt.subplot(222)
plt.imshow(grad_x, cmap='gray')
plt.subplot(223)
plt.imshow(grad_y, cmap='gray')
plt.subplot(224)
plt.imshow(grad_norm, cmap='gray')
plt.show()
```
