import cv2
import numpy as np
from collections import defaultdict

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")
    img = cv2.resize(img, (800, 800))  # Resize for consistency
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce noise
    return img

def detect_board(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Find the largest contour (assuming it's the board)
    board_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(board_contour)
    return (x, y, w, h)

def extract_grid(image, board_coords):
    x, y, w, h = board_coords
    board = image[y:y+h, x:x+w]
    
    # Assuming a 5x5 grid
    cell_h, cell_w = h // 5, w // 5
    tiles = []
    for i in range(5):
        for j in range(5):
            tile = board[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            tiles.append(tile)
    return tiles

def recognize_tiles(tile_images):
    terrain_colors = {
        'Wheat': (0, 255, 255),    # Yellow
        'Forest': (0, 128, 0),     # Green
        'Water': (255, 0, 0),      # Blue
        'Grassland': (0, 255, 0),  # Light Green
        'Swamp': (128, 0, 128),    # Purple
        'Mine': (128, 128, 128)    # Gray
    }

    recognized_board = []
    
    for tile in tile_images:
        # Recognize terrain type based on dominant color
        average_color = np.mean(tile, axis=(0, 1))
        terrain_type = min(terrain_colors, key=lambda x: np.linalg.norm(np.array(terrain_colors[x]) - average_color))
        
        # Count crowns (simplified method - you might need a more robust approach)
        gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_tile, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crown_count = len([c for c in contours if 100 < cv2.contourArea(c) < 500])
        
        recognized_board.append((terrain_type, int(crown_count)))  # Ensure crown_count is an integer
    
    return np.array(recognized_board).reshape(5, 5, 2)

def calculate_score(board_state):
    total_score = 0
    visited = set()

    def dfs(i, j, terrain):
        if (i, j) in visited or i < 0 or i >= 5 or j < 0 or j >= 5 or board_state[i][j][0] != terrain:
            return 0, 0
        
        visited.add((i, j))
        size, crowns = 1, int(board_state[i][j][1])  # Ensure crowns is an integer
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            s, c = dfs(i + di, j + dj, terrain)
            size += s
            crowns += c
        
        return size, crowns

    for i in range(5):
        for j in range(5):
            if (i, j) not in visited:
                size, crowns = dfs(i, j, board_state[i][j][0])
                total_score += size * crowns

    return total_score

def main(image_path):
    try:
        image = preprocess_image(image_path)
        board_coords = detect_board(image)
        tile_images = extract_grid(image, board_coords)
        board_state = recognize_tiles(tile_images)
        score = calculate_score(board_state)
        print(f"Total score: {score}")

        # Visualize the recognized board
        for i in range(5):
            for j in range(5):
                print(f"{board_state[i, j, 0]}({board_state[i, j, 1]})", end=" ")
            print()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    image_path = "4.jpg"
    main(image_path)