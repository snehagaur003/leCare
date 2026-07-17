import cv2 
import numpy as np 

image = cv2.imread("image.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = np.array([5, 50, 50])
upper = np.array([30, 255, 255])

mask = cv2.inRange(
    hsv, 
    lower, 
    upper
)

diseased_pixels = cv2.countNonZero(mask)
total_pixels = image.shape[0]*image.shape[1]
severity = (
    diseased_pixels/total_pixels
)*100
print(
    f"Estimated severity: {severity:.2f}%"
)