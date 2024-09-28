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

maskhav_s = cv.medianBlur(maskhav, ksize = 3)
maskmark_s = cv.medianBlur(maskmark, ksize = 3)
maskskov_s = cv.medianBlur(maskskov, ksize = 3)


# Find contours in the blue mask (hav)
contours, hierarchy = cv.findContours(maskhav, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the mask
maskhav_contours = np.zeros_like(maskhav)  # Create an empty mask to draw contours
cv.drawContours(maskhav_contours, contours, -1, (255, 255, 255), 2)  # White color contours



# Find contours in the blue mask (hav)
contours, hierarchy = cv.findContours(maskmark_s, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the mask
maskmark_contours = np.zeros_like(maskmark_s)  # Create an empty mask to draw contours
cv.drawContours(maskmark_contours, contours, -1, (255, 255, 255), 2)  # White color contours





# Find contours in the blue mask (hav)
contours, hierarchy = cv.findContours(maskskov_s, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the mask
maskskov_contours = np.zeros_like(maskskov_s)  # Create an empty mask to draw contours
cv.drawContours(maskskov_contours, contours, -1, (255, 255, 255), 2)  # White color contours


cv.imshow("test ", maskhav_s)  # type: ignore


# Display the results
cv.imshow("Hav with Contours", maskhav_contours)  # type: ignore
cv.imshow("Mark with Contours", maskmark_contours)  # type: ignore
cv.imshow("Skov with Contours", maskskov_contours)  # type: ignore

cv.waitKey(0)  # Wait for a key press
cv.destroyAllWindows()  # Close all windows