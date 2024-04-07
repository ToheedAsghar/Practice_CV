# Rotation matrix code

![image](https://github.com/ToheedAsghar/Practice_CV/assets/121859513/04ae9f4d-1dd5-42a5-af57-880818b50eda)


```python
import cv2 as cv
import numpy as np

newImage = np.zeros((500, 500), np.uint8)

newImage[100:200, 100:200] = 255

angle = np.deg2rad(45)

center_x = (100 + 200) // 2
center_y = (100 + 200) // 2

for i in range(100, 200):
    for j in range(100, 200):
        x = i - center_x
        y = j - center_y

        new_x = int(x * np.cos(angle) - y * np.sin(angle) + center_x)
        new_y = int(x * np.sin(angle) + y * np.cos(angle) + center_y)

        newImage[new_x, new_y] = newImage[i, j]

cv.imshow("new image", newImage)
cv.waitKey(0)

```

In this code:

- We calculate the rotation matrix for a 2D rotation by 30 degrees.
- We define the center of rotation as the center of the rectangle.
- We iterate over all pixels in the rectangle, translate them to the center of rotation, apply the rotation, and copy the pixel value to the rotated position.
- Finally, we display the rotated image using cv.imshow().

## Output

![image](https://github.com/ToheedAsghar/Practice_CV/assets/121859513/082e8f5a-9e49-47ed-9d1c-f3366ae800d5)
