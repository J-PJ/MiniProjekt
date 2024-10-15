import cv2 as cv
import numpy as np
import os

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

        # Highlight the crowns on the search image
        for (x, y, w, h) in final_crowns:
            cv.rectangle(search_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle

        return search_image, final_crowns

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
    search_image = cv.imread(r'C:\Users\jacob\Downloads\King Domino dataset\King Domino dataset\Cropped and perspective corrected boards\4.jpg')
    if search_image is None:
        raise ValueError("Error: Could not open or find the search image.")

    # Find crowns in the search image
    result_image, crown_coordinates = crown_temps.find_crowns(search_image)

    # Print the coordinates of found crowns
    for i, (x, y, w, h) in enumerate(crown_coordinates):
        print(f"Crown {i + 1}: Top-left: ({x}, {y}), Width: {w}, Height: {h}")

    # Display the result
    cv.imshow("Detected Crowns", result_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

except ValueError as e:
    print(e)
