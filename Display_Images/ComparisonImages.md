# Comparison of Two Images

## Algorithm

- Check if the dimensions of both images match.
- Get the RGB values of both images.
- Calculate the difference in two corresponding pixels of three color components.
- Repeat Steps 2-3 for each pixel of the images.
- Lastly, calculate the percentage by dividing the sum of differences by the number of pixels.


```python

import cv2 as cv
import numpy as np

img1 = cv.imread('huzaifa.png')
img2 = cv.imread('cat.jpeg')

img2 = cv.resize(img2, (759, 617))

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

if h1 != h2 or w1 != w2:
    print("Error: Images Dimensions Mismatch!")
else:
    diff = cv.absdiff(img1, img2)
    sum = np.sum(diff)
    totalPixels = h1 * w1 * 3
    percentage = sum / totalPixels

    print(f'Difference Percentage: {percentage}')


```
