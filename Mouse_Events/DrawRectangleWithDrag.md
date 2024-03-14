# Drawing Rectangle with Mouse Drag

Rectangle can drawn using two points i.e. upper-left and bottom right. We'll record the upper-left coordinate as the EVENT_LBUTTONDOWN and the bottom-right coordinate as EVENT_LBUTTONUP.

```python
import cv2 as cv
import numpy as np

# --- variables --- #
drawing = False
ix, iy = -1, -1

# --- drawing rectangle function --- #
def drawRectangle(event, x, y, flags, param):
    global ix, iy, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.rectangle(img, (ix, iy), (x, y), (0, 255, 255), -1)

# --- Image --- #
img = np.zeros((512, 512, 3), dtype=np.int8)
cv.namedWindow('Window')
cv.setMouseCallback('Window', drawRectangle)

while True:
    cv.imshow('Window', img)
    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
```

- using -1 as thickness to fill the rectangle
- displaying the window atleast 10 milliseconds then waiting for the esc key to close the window
- using ix, iy and drawing as global variables so that we can replace the values globally
