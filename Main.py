import cv2 as cv  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt


# Read the image
image = cv.imread("4.jpg", cv.IMREAD_COLOR)  # type: ignore
#Scoreimage = np.zeros(np.shape(image))  # Make a copy of the original image to store the result

# Convert BGR to HSV
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
maskhav = cv.inRange(hsv, (95, 180, 90), (122, 255, 255))#blue
maskmark = cv.inRange(hsv, (25, 150, 80), (35, 255, 255))#gul
maskskov = cv.inRange(hsv, (40, 50, 0), (65, 255, 60))#grøn

maskmark_s = cv.medianBlur(maskmark, ksize = 3)
maskskov_s = cv.medianBlur(maskskov, ksize = 3)

cv.imshow("Hav", maskhav)  # type: ignore
cv.imshow("Mark", maskmark_s)  # type: ignore
cv.imshow("Skov", maskskov_s)  # type: ignore
cv.waitKey(0)  # type: ignore
cv.destroyAllWindows()  # Close the window after a key press
