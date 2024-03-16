# Kernals

## Blurring Using Kernals

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolution(img, kernel):
    (ih, iw) = img.shape[:2]
    (kh, kw) = kernel.shape[:2]
    
    pad = (kw - 1) // 2
    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output_image = np.zeros((ih, iw, 3), dtype='uint8')

    for y in range(pad, ih + pad):
        for x in range(pad, iw + pad):
            roi = img_padded[y - pad: y + pad + 1, x - pad: x + pad + 1]
            k = np.sum(roi * kernel, axis=(0, 1))
            output_image[y - pad, x - pad] = k
            
    return output_image

img = cv2.imread('Images/Huzaifa.png')
kernel = np.ones((3, 3), np.float32) / 9

blurred_img = convolution(img, kernel)

cv2.imshow('Original Image', img)
cv2.imshow('Blurred Image', blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Kernals

```python
meanBlur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
GaussianBlur = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.250, 0.125], [0.0625, 0.125, 0.0625]])
```

## Inbuilt-Blur Functions

**GaussianBlur**
> blur = cv.GaussianBlur(img,(5,5),0)

**Median Blurring**
> median = cv.medianBlur(img,5)

**Bilateral Filtering**
> blur = cv.bilateralFilter(img,9,75,75)

