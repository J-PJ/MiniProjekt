import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

class KingdominoAnalyzer:
    def __init__(self, crown_template_paths):
        self.crown_temps = CrownTemplates(crown_template_paths)

    def analyze_image(self, image_path):
        # Read the image
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Error: Could not open or find the image at {image_path}")

        # Perform terrain analysis
        terrain_matrix = self.analyze_terrain(image)

        # Find crowns
        image_with_crowns, crown_coordinates = self.crown_temps.find_crowns(image)

        return terrain_matrix, image_with_crowns, crown_coordinates

    def analyze_terrain(self, image):
        # Convert BGR to HSV
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Create masks
        masks = {
            'hav': cv.inRange(hsv, (95, 170, 100), (122, 255, 255)),  # blue
            'mark': cv.inRange(hsv, (20, 200, 185), (29, 255, 255)),  # yellow
            'skov': cv.inRange(hsv, (40, 50, 0), (65, 255, 60)),  # green
            'grass': cv.inRange(hsv, (35, 0, 120), (60, 255, 255)),  # green
            'wastland': cv.inRange(hsv, (19, 40, 0), (28, 190, 155)),  # brown
            'mine': cv.inRange(hsv, (0, 0, 0), (255, 255, 40)),  # sort
            'Start': cv.inRange(hsv, (80, 0, 0), (120, 190, 255)),  # brown blue
            'bord': cv.inRange(hsv, (19, 40, 135), (28, 190, 255)),  # brown
        }

        # Process masks
        processed_masks = {name: self.process_mask(mask) for name, mask in masks.items()}

        # Create 5x5 grid
        grid = self.create_grid(image)

        # Analyze each grid square
        matrix = []
        for i in range(5):
            row = []
            for j in range(5):
                x1, y1, x2, y2 = grid[i*5 + j]
                square_type = self.analyze_grid_square(x1, y1, x2, y2, processed_masks)
                row.append(square_type)
            matrix.append(row)

        return matrix

    def process_mask(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv.erode(mask, kernel, iterations=1)
        dilated = cv.dilate(eroded, kernel, iterations=1)
        return dilated

    def create_grid(self, image, grid_size=5):
        height, width = image.shape[:2]
        grid_h, grid_w = height // grid_size, width // grid_size
        return [(i*grid_h, j*grid_w, (i+1)*grid_h, (j+1)*grid_w) 
                for i in range(grid_size) for j in range(grid_size)]

    def analyze_grid_square(self, x1, y1, x2, y2, masks):
        results = []
        for name, mask in masks.items():
            count = np.sum(mask[y1:y2, x1:x2]) // 255
            total = (y2-y1) * (x2-x1)
            percentage = count / total
            results.append((name, percentage))
        
        max_result = max(results, key=lambda x: x[1])
        if max_result[1] < 0.01:  # Check if the highest percentage is below 1%
            return "error"
        else:
            return max_result[0]

class CrownTemplates:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.template_groups = []
        self.create_template_groups()

    def rotate_image(self, image, angle):
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
        for path in self.image_paths:
            if not os.path.isfile(path):
                raise ValueError(f"Error: File not found at {path}")

            image = cv.imread(path)
            if image is None:
                raise ValueError(f"Error: Could not open or find the image at {path}")

            rotated_images = [self.rotate_image(image, angle) for angle in [0, 90, 180, 270]]
            self.template_groups.append(rotated_images)

    def non_max_suppression(self, boxes, overlapThresh):
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
        region = image[y:y+h, x:x+w]
        hsv_region = cv.cvtColor(region, cv.COLOR_BGR2HSV)
        
        lower_yellow = np.array([20, 100, 10])
        upper_yellow = np.array([30, 255, 255])
        
        mask = cv.inRange(hsv_region, lower_yellow, upper_yellow)
        
        yellow_ratio = np.sum(mask) / (w * h * 255)
        
        return yellow_ratio > 0.5

    def find_crowns(self, search_image):
        found_crowns = []
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
                    if not self.is_yellow(search_image, x, y, w, h):
                        found_crowns.append((x, y, w, h))

        final_crowns = self.non_max_suppression(found_crowns, 0.2)

        for (x, y, w, h) in final_crowns:
            cv.rectangle(search_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return search_image, final_crowns

def main():
    crown_template_paths = [
        'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGForest1.jpg',
        'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGGrass1.jpg',
        'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGMine1.jpg',
        'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGOcean1.jpg',
        'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGWaste1.jpg',
        'C:/Users/jacob/Downloads/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/OGWheat1.jpg'
    ]

    analyzer = KingdominoAnalyzer(crown_template_paths)

    image_path = r'C:\Users\jacob\Downloads\King Domino dataset\King Domino dataset\Cropped and perspective corrected boards\4.jpg'
    
    try:
        terrain_matrix, image_with_crowns, crown_coordinates = analyzer.analyze_image(image_path)

        # Print terrain matrix
        print("Terrain Matrix:")
        for row in terrain_matrix:
            print(row)

        # Print crown coordinates
        print("\nCrown Coordinates:")
        for i, (x, y, w, h) in enumerate(crown_coordinates):
            print(f"Crown {i + 1}: Top-left: ({x}, {y}), Width: {w}, Height: {h}")

        # Display the result
        plt.imshow(cv.cvtColor(image_with_crowns, cv.COLOR_BGR2RGB))
        plt.title("Detected Crowns and Terrain")
        plt.axis('off')
        plt.show()

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()