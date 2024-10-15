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
wastland = cv.inRange(hsv, (19, 40, 0), (28, 190, 155))  # brown
mine = cv.inRange(hsv, (0, 0, 0), (255, 255, 40))  # sort
start = cv.inRange(hsv, (80, 0, 0), (120, 190, 255))  # brown blue
bord = cv.inRange(hsv, (19, 40, 135), (28, 190, 255))  # brown

# Define kernel for erosion and dilation
kernal_size = 5
kernel = np.ones((kernal_size,kernal_size), np.uint8)

# useing opening to get a clean marsk
def process_mask(mask):
    eroded = cv.erode(mask, kernel, iterations=1)
    dilated = cv.dilate(eroded, kernel, iterations=1)
    return dilated

dilated_hav = process_mask(maskhav)
dilated_mark = process_mask(maskmark)
dilated_skov = process_mask(maskskov)
dilated_grass = process_mask(Grassland)
dilated_wastland = process_mask(wastland)
dilated_mine = process_mask(mine)
dilated_start = process_mask(start)
dilated_bord = process_mask(bord)



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
    
    max_result = max(results, key=lambda x: x[1])
    if max_result[1] < 0.01:  # Check if the highest percentage is below 1% to check for erros
        return "error"
    else:
        return max_result[0]

# Create masks dictionary
masks = {
    'hav': dilated_hav,
    'mark': dilated_mark,
    'skov': dilated_skov,
    'grass': dilated_grass,
    'wastland': dilated_wastland,
    'mine': dilated_mine,
    'Start': dilated_start,
    'bord': dilated_bord,
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


def count_points(matrix):
    def dfs(i, j, type, visited):
        if (i < 0 or i >= len(matrix) or 
            j < 0 or j >= len(matrix[0]) or 
            visited[i][j] or 
            matrix[i][j] != type):
            return 0
        
        visited[i][j] = True
        count = 1
        
        # Check all 4 directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for di, dj in directions:
            count += dfs(i + di, j + dj, type, visited)
        
        return count

    visited = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    points = {type: [] for type in set(sum(matrix, []))}
    
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if not visited[i][j]:
                type = matrix[i][j]
                group_size = dfs(i, j, type, visited)
                if group_size > 0: 
                    points[type].append(group_size)
    
    return points

# Use the function
point_counts = count_points(matrix)

# Print the results
for type, groups in point_counts.items():
    print(f"{type}: {groups} - Total: {sum(groups)}")



cv.imshow("Grid", grid_image)
#cv.imshow("test",wastland)

# Display the results
#cv.imshow("Hav", dilated_hav)
#cv.imshow("Mark", dilated_mark)
#cv.imshow("Skov", dilated_skov)
#cv.imshow("Grass", dilated_grass)
#cv.imshow("wasteland",dilated_wastland)
#cv.imshow("Mine", dilated_mine)
#cv.imshow("start", start)
#cv.imshow("start", dilated_start)

cv.waitKey(0)  # Wait for a key press
cv.destroyAllWindows()  # Close all windows