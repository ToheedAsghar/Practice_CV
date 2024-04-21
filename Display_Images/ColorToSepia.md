# Colored Image to Sepia Image Conversion

## Algorithm

- Get the RGB value of the pixel.
- Calculate newRed, new green, newBlue using the above formula (Take the integer value)
- Set the new RGB value of the pixel as per the following condition: 
  - If newRed > 255 then R = 255 else R = newRed
  - If newGreen > 255 then G = 255 else G = newGreen
  - If newBlue > 255 then B = 255 else B = newBlue
- Replace the value of R, G, and B with the new value that we calculated for the pixel.]
- Repeat Step 1 to Step 4 for each pixel of the image.

```python

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def sepia(image):
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])

    row, col = image.shape[:2]
    img = np.zeros_like(image)

    for i in range(row):
        for j in range(col):
            b, g, r = image[i, j]
            red = int(r * 0.393 + 0.796 * g + b * 0.189)
            green = int(0.349 * r + 0.686 * g + 0.168 * b)
            blue = int(0.272 * r + 0.543 * g + 0.131 * b)

            img[i, j] = (min(255, blue), min(255, green), min(red, 255))

    return img


def main() -> None:
    img = cv.imread('huzaifa.png')

    cv.imshow('', img)
    cv.waitKey()

    sepiaImage = sepia(img)

    cv.imshow('', sepiaImage)
    cv.waitKey()


if __name__ == '__main__':
    main()


```
