import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

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
    
matrix = matrix[::-1]
rotated_matrix = list(zip(*matrix[::-1]))

# Udskriv den resulterende matrix
print("Roteret Matrix:")
for row in rotated_matrix:
    print(row)

# visualize the grid on the original image
grid_image = image.copy()
for x1, y1, x2, y2 in grid:
     cv.rectangle(grid_image, (y1, x1), (y2, x2), (0, 255, 0), 2)




class Crown_Temps:
    def __init__(self, image_paths):
        """Initialize with a list of image file paths."""
        self.image_paths = image_paths
        self.template_groups = []  # List to store grouped images
        self.create_template_groups()

    def rotate_image(self, image, angle):
        """Rotate the image by the specified angle."""
        if angle == 0:
            return image
        elif angle == 90:
            return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv.rotate(image, cv.ROTATE_180)
        elif angle == 270:
            return cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise ValueError("Angle must be one of: 0, 90, 180, 270.")

    def create_template_groups(self):
        """Create lists of original and rotated images for each template."""
        for path in self.image_paths:
            if not os.path.isfile(path):
                raise ValueError(f"Error: File not found at {path}")

            image = cv.imread(path)
            if image is None:
                raise ValueError(f"Error: Could not open or find the image at {path}")

            rotated_images = [self.rotate_image(image, angle) for angle in [0, 90, 180, 270]]
            self.template_groups.append(rotated_images)

    def non_max_suppression(self, boxes, overlapThresh):
        """Perform non-maximum suppression to eliminate overlapping bounding boxes."""
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].tolist()

    def is_yellow(self, image, x, y, w, h):
        """Check if the region is predominantly yellow."""
        region = image[y:y+h, x:x+w]
        hsv_region = cv.cvtColor(region, cv.COLOR_BGR2HSV)
        
        # Define range for yellow color in HSV
        lower_yellow = np.array([20, 100, 10])
        upper_yellow = np.array([30, 255, 255])
        
        # Create a mask for yellow color
        mask = cv.inRange(hsv_region, lower_yellow, upper_yellow)
        
        # Calculate the percentage of yellow pixels
        yellow_ratio = np.sum(mask) / (w * h * 255)
        
        # If more than 50% of the region is yellow, consider it a yellow piece
        return yellow_ratio > 0.5

    def find_crowns(self, search_image):
        """Find crowns in the search image using template matching and color filtering."""
        found_crowns = []
        height, width, _ = search_image.shape

        # Initialize a 5x5 matrix for counting crowns
        crowns_count_matrix = np.zeros((5, 5), dtype=int)

        for template_group in self.template_groups:
            for template in template_group:
                template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
                search_image_gray = cv.cvtColor(search_image, cv.COLOR_BGR2GRAY)

                result = cv.matchTemplate(search_image_gray, template_gray, cv.TM_CCOEFF_NORMED)
                threshold = 0.7
                loc = np.where(result >= threshold)

                h, w = template_gray.shape
                for pt in zip(*loc[::-1]):
                    x, y = pt
                    # Check if the detected region is yellow
                    if not self.is_yellow(search_image, x, y, w, h):
                        found_crowns.append((x, y, w, h))

        # Apply non-maximum suppression
        final_crowns = self.non_max_suppression(found_crowns, 0.2)

        # Calculate cell dimensions
        cell_width = width // 5
        cell_height = height // 5

        # Update the crowns count matrix
        for (x, y, w, h) in final_crowns:
            # Determine which cell the crown belongs to
            cell_x = min(x // cell_width, 4)  # Ensure it does not exceed matrix bounds
            cell_y = min(y // cell_height, 4)  # Ensure it does not exceed matrix bounds
            crowns_count_matrix[cell_y, cell_x] += 1  # Increment the count

        # Highlight the crowns on the search image
        for (x, y, w, h) in final_crowns:
            cv.rectangle(search_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle

        return search_image, crowns_count_matrix  # Return the search image and the crowns count matrix

    def display_images(self):
        """Display all images in each group."""
        for i, images in enumerate(self.template_groups):
            for j, img in enumerate(images):
                cv.imshow(f"Template {i + 1} - Rotation {j * 90} degrees", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

# Usage Example
image_paths = [
    'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGForest1.jpg',
    'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGGrass1.jpg',
    'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGMine1.jpg',
    'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGOcean1.jpg',
    'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGWaste1.jpg',
    'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGWheat1.jpg'
]

try:
    crown_temps = Crown_Temps(image_paths)

    # Load the search image
    
    if image is None:
        raise ValueError("Error: Could not open or find the search image.")

    # Find crowns in the search image
    result_image, crowns_count_matrix = crown_temps.find_crowns(image)

    # Print the 5x5 matrix of crown counts
    print("Crown Count Matrix:")
    print(crowns_count_matrix)

    # Display the result
    cv.imshow("Detected Crowns", result_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

except ValueError as e:
    print(e)















def count_points_and_crowns(land_matrix, crown_matrix):
    def dfs(i, j, land_type, visited):
        if (i < 0 or i >= len(land_matrix) or 
            j < 0 or j >= len(land_matrix[0]) or 
            visited[i][j] or 
            land_matrix[i][j] != land_type):
            return 0, 0
        
        visited[i][j] = True
        group_size = 1
        crown_count = crown_matrix[i][j]  # Count the crowns in the current cell
        
        # Check all 4 directions (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for di, dj in directions:
            size, crowns = dfs(i + di, j + dj, land_type, visited)
            group_size += size
            crown_count += crowns
        
        return group_size, crown_count

    visited = [[False for _ in range(len(land_matrix[0]))] for _ in range(len(land_matrix))]
    points = {land_type: [] for land_type in set(sum(land_matrix, []))}

    for i in range(len(land_matrix)):
        for j in range(len(land_matrix[0])):
            if not visited[i][j]:
                land_type = land_matrix[i][j]
                group_size, total_crowns = dfs(i, j, land_type, visited)
                if group_size > 0: 
                    points[land_type].append((group_size, total_crowns))
    
    return points

# Use the new function with the tile type matrix and the crown count matrix
points_with_crowns = count_points_and_crowns(matrix, crowns_count_matrix)

# Print the results for both land type groups and crown counts
for land_type, groups in points_with_crowns.items():
    group_sizes = [group[0] for group in groups]
    crown_counts = [group[1] for group in groups]
    print(f"{land_type}: Groups Sizes: {group_sizes}, Crowns: {crown_counts} - Total Points: {sum(group_size * crown for group_size, crown in groups)}")

cv.waitKey(0)  # Wait for a key press
cv.destroyAllWindows()  # Close all windows