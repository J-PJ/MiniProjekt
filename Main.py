import cv2 as cv  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt


# Read the image
image = cv.imread("4.jpg", cv.IMREAD_COLOR)  # type: ignore
#Scoreimage = np.zeros(np.shape(image))  # Make a copy of the original image to store the result

# Convert BGR to HSV
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)


maskhav = cv.inRange(hsv, (95, 170, 100), (122, 255, 255))#blue
maskmark = cv.inRange(hsv, (20, 200, 185), (29, 255, 255))#gul
maskskov = cv.inRange(hsv, (40, 50, 0), (65, 255, 60))#grøn
Grassland = cv.inRange(hsv, (35, 0, 0), (45, 255, 250))#grøn
mine = cv.inRange(hsv, (0, 0, 0), (35, 105, 255))#grøn

maskhav_s = cv.medianBlur(maskhav, ksize = 3)
maskmark_s = cv.medianBlur(maskmark, ksize = 3)
maskskov_s = cv.medianBlur(maskskov, ksize = 3)






# Display the results
#cv.imshow("Hav", maskhav_s)  # type: ignore
#cv.imshow("Mark", maskmark_s)  # type: ignore
#cv.imshow("Skov", maskskov_s)  # type: ignore
cv.imshow("Græs", Grassland)  # type: ignore
cv.imshow("Mine", mine)  # type: ignore

cv.waitKey(0)  # Wait for a key press
cv.destroyAllWindows()  # Close all windows