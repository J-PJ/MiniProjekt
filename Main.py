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
start = cv.inRange(hsv, (80, 0, 0), (120, 190, 255))  # green

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
dilated_start = process_mask(start)

def create_grid(image, grid_size=5):
    height, width = image.shape[:2]
    grid_h, grid_w = height // grid_size, width // grid_size
    return [(i*grid_h, j*grid_w, (i+1)*grid_h, (j+1)*grid_w) 
            for i in range(grid_size) for j in range(grid_size)]

def analyze_grid_square(x1, y1, x2, y2, masks):
    results = []
    for name, mask in masks.items():
        count = np.sum(mask[y1:y2, x1:x2]) // 255
        total = (y2-y1) * (x2-x1)
        percentage = count / total
        results.append((name, percentage))
    return max(results, key=lambda x: x[1])[0]

# Create masks dictionary
masks = {
    'hav': dilated_hav,
    'mark': dilated_mark,
    'skov': dilated_skov,
    'grass': dilated_grass,
    'mine': dilated_mine,
    'Start': dilated_start
}

# Create 5x5 grid
grid = create_grid(image)

# Analyze each grid square
matrix = []
for i in range(5):
    row = []
    for j in range(5):
        x1, y1, x2, y2 = grid[i*5 + j]
        square_type = analyze_grid_square(x1, y1, x2, y2, masks)
        row.append(square_type)
    matrix.append(row)

# Print the resulting matrix
for row in matrix:
    print(row)

# Optionally, visualize the grid on the original image
grid_image = image.copy()
for x1, y1, x2, y2 in grid:
     cv.rectangle(grid_image, (y1, x1), (y2, x2), (0, 255, 0), 2)


def count_points():



cv.imshow("Grid", grid_image)

# Display the results
# cv.imshow("Hav", dilated_hav)
# cv.imshow("Mark", dilated_mark)
# cv.imshow("Skov", dilated_skov)
# cv.imshow("Grass", dilated_grass)
# cv.imshow("Mine", dilated_mine)
#cv.imshow("start", start)
#cv.imshow("start", dilated_start)

cv.waitKey(0)  # Wait for a key press
cv.destroyAllWindows()  # Close all windows