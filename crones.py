import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv.imread("4.jpg", cv.IMREAD_COLOR)

# Convert BGR to HSV
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Create masks
maskkrone = cv.inRange(hsv, (23, 0, 100), (35, 100, 255))  # blue


# Define kernel for erosion and dilation
kernal_size = 5
kernel = np.ones((kernal_size,kernal_size), np.uint8)

# useing opening to get a clean marsk
def process_mask(mask):
    eroded = cv.erode(mask, kernel, iterations=1)
    dilated = cv.dilate(eroded, kernel, iterations=1)
    return dilated

dilated_krone = process_mask(maskkrone)


# making a 5*5 grid
def create_grid(image, grid_size=5):
    height, width = image.shape[:2]
    grid_h, grid_w = height // grid_size, width // grid_size
    return [(i*grid_h, j*grid_w, (i+1)*grid_h, (j+1)*grid_w) 
            for i in range(grid_size) for j in range(grid_size)]

# seeing what type there is in eage grid
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
    'krone': dilated_krone,
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

# visualize the grid on the original image
grid_image = image.copy()
for x1, y1, x2, y2 in grid:
     cv.rectangle(grid_image, (y1, x1), (y2, x2), (0, 255, 0), 2)



cv.imshow("start", maskkrone)

cv.waitKey(0)  # Wait for a key press
cv.destroyAllWindows()  # Close all windows