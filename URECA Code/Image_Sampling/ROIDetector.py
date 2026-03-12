"""
Phase 1: ROI Detector
======================
Pipeline:
  1. Horizon Mask   → Canny + Hough → discard rows above the horizon line
  2. Grid Tessellation → divide water region + 1 block above horizon into SxS blocks
  3. Dynamic Baseline  → compute median HSV of every block → "what water looks like"
  4. Anomaly Detection → flag blocks whose mean HSV deviates > threshold from baseline
  5. Spatial Validation→ group contiguous flagged blocks; drop isolated singles (noise) Neghbour (Maybe include and or edge detection)
  6. ROI Crop          → return bounding-box crops + (x, y, w, h) tuples to Phase 2

Input : raw left-camera BGR image (np.ndarray, shape H x W x 3 HSV)
Output: list of (crop_bgr, (x, y, w, h)) tuples
"""

import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt

class ROIConfig:

    #Call this with config = ROIConfig(img) 
    def __init__(
        self,
        image=None,
        canny_low=50,
        canny_high=150,
        hough_rho=1.0,
        hough_theta=np.pi / 180,
        hough_threshold=50,
        hough_min_line_length=None, #Set below 
        hough_max_line_gap=50,
        horizon_angle_tol_deg=15.0,
        grid_size=32, #For grid tesslation
        hue_thresh=40.0, #For Anomaly detection 
        sat_thresh=40.0,
        val_thresh=40.0,
        min_contiguous_blocks=2, #For Neighbour mnatching
        min_crop_px=16
    ):
        # 1. Handle Image Dimensions
        self.img_width = None
        self.img_height = None
        
        if image is not None:
            # image.shape is (height, width, channels)
            self.img_height, self.img_width = image.shape[:2]
            
            # Auto-calculate hough_min_line_length if not explicitly provided
            if hough_min_line_length is None:
                hough_min_line_length = int(self.img_width * 0.3)
        
        # Fallback if no image and no value provided
        if hough_min_line_length is None:
            hough_min_line_length = 100  
        
        # Horizon detection
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold #votes for a line
        self.hough_min_line_length = hough_min_line_length #Adjust this to width * 0.5 for long lines only
        self.hough_max_line_gap = hough_max_line_gap #To make it easier for lines
        self.horizon_angle_tol_deg = horizon_angle_tol_deg

        # Grid tessellation
        self.grid_size = grid_size

        # Anomaly detection
        self.hue_thresh = hue_thresh
        self.sat_thresh = sat_thresh
        self.val_thresh = val_thresh

        # Spatial validation
        self.min_contiguous_blocks = min_contiguous_blocks

        # Minimum crop size to pass to RCE
        self.min_crop_px = min_crop_px

class ROIDetector: 
    def __init__(self, config: ROIConfig = None):
        self.config = config if config is not None else ROIConfig()

    #Step 1: Horizon mask: np.array --> y:axis int (Anything above this will be removed)
    # y = -1 --> there is no horizon line, contiue normal chop & RCE.
    # y:axis --> whichever edge where the line exit is higher 
    # Output canny edge ddetection image for neighbour later ***
    def detect_horizon(self, image: np.ndarray) -> int:
        y = -1 #Output

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, config.canny_low, config.canny_high)

        #Outputs [[x1,y1,x2,y2],[]]
        horizon_lines = cv2.HoughLinesP(
        edges, rho=config.hough_rho, theta=config.hough_theta, threshold=config.hough_threshold, minLineLength=config.hough_min_line_length, maxLineGap=config.hough_max_line_gap,)

        #If got output (since the min_line length is set)
        if horizon_lines is not None and len(horizon_lines) > 0:

            flat_lines = [line[0] for line in horizon_lines]

            longest_line = max(flat_lines, key=lambda p: (p[2] - p[0])**2 + (p[3] - p[1])**2) 
            x1, y1, x2, y2 = longest_line

            #Then use that x,y to find y=mx + c, then max the 2 sides of the image to
            if x2 != x1:
                m = (y2 - y1) / (x2 - x1)  # Slope (m)
                c = y1 - (m * x1)          # Intercept (c)
            
                #max value of y-axis left and right 
                left_y = int(c)
                right_y = int(m * config.img_width + c)
                y = min(left_y, right_y)
                
            else: # Line Horizontal
                y = min(y1,y2)

            return y
        else:
            return y #y is -1, where just skip this and go to chopp all go RCE

    def debug_horizon(self, image: np.ndarray):
        """Visualizer: Plots Original, Canny Edges, and Hough Lines side-by-side."""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.config.canny_low, self.config.canny_high)

        horizon_lines = cv2.HoughLinesP(
            edges, 
            rho=self.config.hough_rho, 
            theta=self.config.hough_theta, 
            threshold=self.config.hough_threshold, 
            minLineLength=self.config.hough_min_line_length, 
            maxLineGap=self.config.hough_max_line_gap
        )

        line_img = image.copy()
        y_max = -1

        if horizon_lines is not None and len(horizon_lines) > 0:
            flat_lines = [line[0] for line in horizon_lines]
            
            for x1, y1, x2, y2 in flat_lines:
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                
            longest_line = max(flat_lines, key=lambda p: (p[2] - p[0])**2 + (p[3] - p[1])**2)
            x1, y1, x2, y2 = longest_line
            
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 4)

            if x2 != x1:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - (m * x1)
                
                left_y = int(c)
                right_y = int(m * self.config.img_width + c)
                y_max = max(left_y, right_y)
                
                cv2.line(line_img, (0, left_y), (self.config.img_width, right_y), (0, 0, 255), 2)
            else:
                y_max = max(y1, y2)

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        line_img_rgb = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title('1. Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title('2. Canny Edge Mask')
        axes[1].axis('off')
        
        axes[2].imshow(line_img_rgb)
        axes[2].set_title(f'3. Hough Lines (Max Y: {y_max})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

    def tessellate_grid(self, image:np.ndarray, size:int = None) -> list[list[tuple[int, int, int, int]]]:
        #Divide image into size blocks of pixel, with coorindates,x ,y width & height 
        #And store it into a 2D array. If i want to invesitgate grid 1, it will be grid[0][0]
        if size is None:
            size = self.config.grid_size
        
        grid = []
        h, w = image.shape[:2]

        # 1. Loop exactly 'num_blocks' times for the rows (Y-axis)
        for row in range(size):
            row_blocks = []
            
            # Calculate exact Y start and end for this row to handle remainders cleanly
            y = int(row * h / size)
            next_y = int((row + 1) * h / size)
            block_h = next_y - y
            
            # 2. Loop exactly 'num_blocks' times for the columns (X-axis)
            for col in range(size):
                
                # Calculate exact X start and end for this column
                x = int(col * w / size)
                next_x = int((col + 1) * w / size)
                block_w = next_x - x
                
                # Add the block (x, y, w, h) to the current row
                row_blocks.append((x, y, block_w, block_h))
            
            # 3. Add the finished row (which has exactly 32 columns) to the grid
            grid.append(row_blocks)

        return grid

    def compute_baseline(self, image: np.ndarray, horizon_y:int) -> np.ndarray:
        #3. Dynamic Baseline -> find the median ** MIGHT CHANGE NEXT TIME
        
        # 1. Convert the entire image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 2. Determine where to slice the image
        if horizon_y != -1:
            # start at horizon_Y + 1 
            start_y = max(0, horizon_y) 
        else:
            start_y = 0 # No horizon, use the whole image

        # 3. Slice the array
        # This grabs everything from start_y down to the bottom, across all X columns
        water_region = hsv_image[start_y:, :]
        
        # 4. Flatten the 2D region into a 1D list of pixels, then find the median
        flat_pixels = water_region.reshape(-1, 3)
        baseline_hsv = np.median(flat_pixels, axis=0)
        
        return baseline_hsv

    # def anomaly_detection(self, image: np.ndarray, grid:list[(int,int,int,int)], horizon_row_index:int, water_baseline:np.ndarray) -> dict[(int,int)]:
    #     #Output List of grids [(0 to size, 0 to size)]

    #     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #     num_rows = len(grid)
    #     num_cols = len(grid[0])
    #     anomalies = []

    #     # Loop through the water rows
    #     for r in range(horizon_row_index, num_rows):
    #         for c in range(num_cols):
    #             x, y, w, h = grid[r][c]
    #             block_pixels = hsv_image[y:y+h, x:x+w]
                
    #             if block_pixels.size > 0:
    #                 # Get mean HSV for this single block [H, S, V] - Axis 1: x, Axis 2: y, Axis 3: HSV
    #                 block_hsv = np.mean(block_pixels, axis=(0, 1))
                    
    #                 # 3. Subtract all 3 channels at once
    #                 delta = np.abs(block_hsv - water_baseline)
                    
    #                 # 4. Extract differences (delta[0] is H, delta[1] is S, delta[2] is V)
    #                 # Using min() to handle the 180-degree Hue circle wrap-around
    #                 h_diff = min(delta[0], 180 - delta[0]) 
    #                 s_diff = delta[1]
    #                 v_diff = delta[2]
                    
    #                 if (h_diff > self.config.hue_thresh) or (s_diff > self.config.sat_thresh) or \
    #                    (v_diff > self.config.val_thresh):
                        
    #                     # Save anomalities and its colour 
    #                     anomalies[(r, c)] = block_hsv

    #     return anomalies

    def anomaly_detection(self, image: np.ndarray, grid: list, water_baseline: np.ndarray) -> dict:
        """
        Scans the ENTIRE image. Compares every block to the water baseline.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        num_rows = len(grid)
        num_cols = len(grid[0])
        anomalies = {}

        # START AT ROW 0! Scan the whole image.
        for r in range(num_rows):
            for c in range(num_cols):
                x, y, w, h = grid[r][c]
                block_pixels = hsv_image[y:y+h, x:x+w]
                
                if block_pixels.size > 0:
                    block_hsv = np.mean(block_pixels, axis=(0, 1))
                    delta = np.abs(block_hsv - water_baseline)
                    
                    h_diff = min(delta[0], 180 - delta[0]) 
                    s_diff = delta[1]
                    v_diff = delta[2]
                    
                    if (h_diff > self.config.hue_thresh) or \
                       (s_diff > self.config.sat_thresh) or \
                       (v_diff > self.config.val_thresh):
                        
                        anomalies[(r, c)] = block_hsv

        return anomalies
    
    def debug_anomalies(self, image: np.ndarray, grid: list, anomalies_dict: dict):
        """
        Visualizer: Draws the grid and highlights anomaly blocks in red.
        Expects anomalies_dict to be formatted as {(r, c): [H, S, V]}
        """
        # 1. Create a copy of the image so we don't draw on the original
        debug_img = image.copy()

        # 2. Draw the whole grid in faint grey so you can see the tessellation
        for row in grid:
            for x, y, w, h in row:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (200, 200, 200), 1)

        # 3. Highlight the flagged anomalies in thick RED
        # We use .keys() because our coordinates are the keys in the dictionary
        for r, c in anomalies_dict.keys():
            x, y, w, h = grid[r][c]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # 4. Display the result using Matplotlib
        img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(f'Anomaly Detection: {len(anomalies_dict)} Blocks Flagged')
        plt.axis('off')
        plt.show()

    def neighbour_checker(self, image: np.ndarray, grid: list):
        pass

    def spatial_validation(self, anomalies_dict: dict) -> list[list[tuple[int, int]]]:
        visited = set()
        valid_objects = []
        
        # 8-way directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        color_tolerance = 20.0 # How much the Hue can change before we consider it a different object

        for start_coord, start_hsv in anomalies_dict.items():
            if start_coord in visited:
                continue
                
            current_object_coords = []
            queue = [start_coord]
            visited.add(start_coord)
            
            # The exact color of the first block in this object
            seed_hue = start_hsv[0]

            while queue:
                curr_r, curr_c = queue.pop(0)
                current_object_coords.append((curr_r, curr_c))
                
                for dr, dc in directions:
                    neighbor = (curr_r + dr, curr_c + dc)
                    
                    if neighbor in anomalies_dict and neighbor not in visited:
                        neighbor_hue = anomalies_dict[neighbor][0]
                        
                        # Calculate Hue difference
                        hue_diff = min(abs(seed_hue - neighbor_hue), 180 - abs(seed_hue - neighbor_hue))
                        
                        # ONLY group it if it matches the color! This stops the trees.
                        if hue_diff <= color_tolerance:
                            visited.add(neighbor)
                            queue.append(neighbor)
            
            # Filters
            if len(current_object_coords) < self.config.min_contiguous_blocks:
                continue
                
            columns = [c for r, c in current_object_coords]
            object_width = max(columns) - min(columns) + 1
            
            # Delete wide objects (the trees)
            if object_width > 6:
                continue
                
            valid_objects.append(current_object_coords)

        return valid_objects

    def debug_spatial_validation(self, image: np.ndarray, grid: list, valid_objects: list):
        """
        Visualizer: Draws green bounding boxes around the final surviving objects.
        """
        debug_img = image.copy()

        # Faintly draw the grid for reference
        for row in grid:
            for x, y, w, h in row:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (200, 200, 200), 1)

        # Draw green bounding boxes around valid objects
        for i, obj_coords in enumerate(valid_objects):
            # Extract rows and columns
            rows = [r for r, c in obj_coords]
            cols = [c for r, c in obj_coords]
            
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            
            # Top-Left Pixel Coordinate
            x1 = grid[min_r][min_c][0]
            y1 = grid[min_r][min_c][1]
            
            # Bottom-Right Pixel Coordinate
            x_br, y_br, w_br, h_br = grid[max_r][max_c]
            x2 = x_br + w_br
            y2 = y_br + h_br
            
            # Draw the box and add a label
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(debug_img, f"Target {i+1}", (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display result
        img_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(f'Phase 1 Output: {len(valid_objects)} Valid Targets Found')
        plt.axis('off')
        plt.show()

    def process(self, image: np.ndarray):
        #Run the whole ROIDetection

        #1. Detect Horizon
        horizon_y = self.detect_horizon(image)

        if horizon_y != -1:
            horizon_row_index = max(0, (horizon_y // self.config.grid_size) - 1) 
        else:
            horizon_row_index = 0

        #2. Split up the image to get (x,y,w,h)
        grid = self.tessellate_grid(image)


image_path = os.path.join(os.getcwd(), "Test3.jpg")
img = cv2.imread(image_path)
# cv2.imshow("img",img)
# cv2.waitKey(0)
if img is None:
    print("ERROR! NO IMG FOUND!")
    exit()
config = ROIConfig(img)

roi_detector = ROIDetector(config)

# 3. Call the methods directly on the instance
horizon_y = roi_detector.detect_horizon(img)
if horizon_y != -1:
    horizon_row_index = max(0, (horizon_y // (config.img_height // roi_detector.config.grid_size)) - 1) 
else:
    horizon_row_index = 0

print(horizon_y)
print(horizon_row_index)

roi_detector.debug_horizon(img)
# 4. Run the debugger
# Run Step 2: Grid
grid = roi_detector.tessellate_grid(img)

# Run Step 3: Baseline
baseline = roi_detector.compute_baseline(img, horizon_y)

# Run Step 4: Anomalies
anomalies = roi_detector.anomaly_detection(img,grid,baseline)

# RUN THE NEW DEBUGGER!
roi_detector.debug_anomalies(img, grid, anomalies)

# Run Step 5
valid_objects = roi_detector.spatial_validation(anomalies)

#debug step 5
roi_detector.debug_spatial_validation(img,grid,valid_objects)