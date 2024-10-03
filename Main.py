import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv.imread("4.jpg", cv.IMREAD_COLOR)

# Convert BGR to HSV
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Create masks
maskhav = cv.inRange(hsv, (95, 170, 100), (122, 255, 255))  # blue
maskmark = cv.inRange(hsv, (20, 200, 185), (29, 255, 255))  # yellow
maskskov = cv.inRange(hsv, (40, 50, 0), (65, 255, 60))  # green
Grassland = cv.inRange(hsv, (35, 0, 120), (60, 255, 255))  # green
mine = cv.inRange(hsv, (19, 40, 0), (28, 190, 255))  # green


kernal_size = 5
# Define kernel for erosion and dilation
kernel = np.ones((kernal_size,kernal_size), np.uint8)

def process_mask(mask):
    eroded = cv.erode(mask, kernel, iterations=1)
    dilated = cv.dilate(eroded, kernel, iterations=1)
    return dilated

dilated_hav = process_mask(maskhav)
dilated_mark = process_mask(maskmark)
dilated_skov = process_mask(maskskov)
dilated_grass = process_mask(Grassland)
dilated_mine = process_mask(mine)

def count_regions(mask):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    # Filter out very small regions (likely noise)
    significant_regions = [i for i in range(1, num_labels) if stats[i, cv.CC_STAT_AREA] > 100]
    return len(significant_regions)

# Count regions for each terrain type
water_count = count_regions(dilated_hav)
field_count = count_regions(dilated_mark)
forest_count = count_regions(dilated_skov)
mine_count = count_regions(dilated_grass)
dilated_mine = count_regions(dilated_mine)

# Print results
print(f"Water regions: {water_count}")
print(f"Field regions: {field_count}")
print(f"Forest regions: {forest_count}")
print(f"Mine regions: {mine_count}")


# Display the results
# cv.imshow("Hav", dilated_hav)
# cv.imshow("Mark", dilated_mark)
# cv.imshow("Skov", dilated_skov)
# cv.imshow("Grass", dilated_grass)
# cv.imshow("Mine", dilated_mine)

cv.waitKey(0)  # Wait for a key press
cv.destroyAllWindows()  # Close all windows