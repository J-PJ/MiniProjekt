import cv2 as cv  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt

# Read the image
image = cv.imread("4.jpg", cv.IMREAD_COLOR)  # type: ignore
#Scoreimage = np.zeros(np.shape(image))  # Make a copy of the original image to store the result

# Convert BGR to HSV
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
maskhav = cv.inRange(hsv, (100, 90, 90), (120, 255, 255))#blue
maskmark = cv.inRange(hsv, (0, 230, 80), (35, 255, 255))#gul
maskskov = cv.inRange(hsv, (0, 80, 0), (50, 255, 60))#grøn


cv.imshow("Hav", maskhav)  # type: ignore
cv.imshow("Mark", maskmark)  # type: ignore
cv.imshow("Skov", maskskov)  # type: ignore
cv.waitKey(0)  # type: ignore
cv.destroyAllWindows()  # Close the window after a key press
