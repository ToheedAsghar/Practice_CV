# Computer Vision Questions/Answers 

1. **How does histogram equalization enhance image contrast?**

Answer: Histogram equalization is a technique used to improve the contrast of an image by redistributing pixel intensities.
It achieves this by stretching the intensity values across the entire range, making the image visually more appealing.
Example: In medical imaging, histogram equalization can be applied to X-ray images to enhance the visibility of specific structures, such as bones or soft tissues.

2. **What is the purpose of the Non-Maximum Suppression (NMS) algorithm in object detection?**

Answer: NMS is used in object detection to eliminate redundant and overlapping bounding boxes, ensuring that only the most confident and accurate predictions are retained.
Example: In a pedestrian detection system, NMS would prevent multiple bounding boxes from being generated for the same pedestrian, improving the precision of the detection.

3. **What is the purpose of the Canny Edge Detector in computer vision?**

Answer: The Canny Edge Detector is used for detecting edges in images. It involves multiple steps, including smoothing the image, finding gradients, and applying non-maximum suppression.
In robotics, the Canny Edge Detector can be employed for detecting boundaries of objects for navigation.

4. **Explain the concept of image denoising and its applications.**

Answer: Image denoising involves removing noise from an image while preserving important details.
This is critical in various applications such as medical imaging, where noise reduction enhances the clarity of diagnostic images.
Denoising algorithms, like the Non-Local Means algorithm, are commonly used for this purpose.

5. **@hat is the purpose of the Harris Corner Detector in feature extraction?**

Answer: The Harris Corner Detector identifies key points (corners) in an image based on variations in intensity.
These corners serve as distinctive features for image matching and object recognition.
In augmented reality applications, Harris corners can be used for tracking and aligning virtual objects with the real-world scene.

6. **What is the purpose of the Sobel operator in image processing?**

Answer: The Sobel operator is used for edge detection by approximating the gradient of the image intensity.
It is particularly effective in highlighting vertical and horizontal edges. In autonomous vehicles, the Sobel operator can be utilized to detect lane boundaries from images captured by cameras.

7. **Explain the concept of image gradient.**

Answer: Image gradient represents the rate of change of pixel values in an image. It is used to identify edges and boundaries within an image, playing a crucial role in edge detection algorithms.
Example: In autonomous vehicles, image gradients can be used to detect lane markings, allowing the vehicle to stay within its lane.
