"""
Phase 1: ROI Detector TO BE UPDATED
======================
  1. Horizon Mask   → Guassian + Sobel filter + threshold -> Hough → discard rows above the horizon line *Store Guassian & Sobel for Step 3 and Phase 2 RCE 
  2. Grid Tessellation → divide water region + 1 block above horizon into SxS blocks
  3. Dynamic Baseline  → compute median HSV of every block → "what water looks like"
  4. Anomaly Detection → flag blocks whose mean HSV deviates > threshold from baseline
  5. Spatial Validation→ group contiguous flagged blocks; drop isolated singles (noise) Neghbour (Maybe include and or edge detection)
  6. ROI Crop          → return bounding-box crops + (x, y, w, h) tuples to Phase 2

Coarse to fine 
Loose grid + fine grid -> run grid tesslation and Anomly detection twice
Anomly detection -> Lower grid size, lower HSV diff threshold (less sensitive)

Input : raw left-camera BGR image (np.ndarray, shape H x W x 3 HSV)
Output: cropped 64 x 64 of guassian & soble filter (x,y,r,g,b)
Targets[
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    gaussian_bgr: np.ndarray         # Shape: (64, 64, 3), dtype: uint8
    sobel_bgr: np.ndarray            # Shape: (64, 64, 3), dtype: uint8 ]
"""

import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np

import numpy as np

class ROIConfig:
    def __init__(
        self,
        image=None,
        mode="sea",  
        
        edges_threshold=None,
        hough_threshold=None,
        hough_max_line_gap=None,
        horizon_angle_tol_deg=None,
        hue_thresh=None,
        sat_thresh=None,
        val_thresh=None,
        
        # --- NEW VARIABLES ---
        block_variance_thresh=None,
        width_filter_divisor=None,
        
        hough_rho=1.0,
        hough_theta=np.pi / 180,
        hough_min_line_length=None,
        grid_size=8,
        min_contiguous_blocks=1,
        min_crop_px=16
    ):
        presets = {
            "sea": {
                "edges_threshold": 60,
                "hough_threshold": 120,
                "hough_max_line_gap": 20,
                "horizon_angle_tol_deg": 5.0,   
                "hue_thresh": 20.0,             
                "sat_thresh": 20.0,
                "val_thresh": 20.0,
                # --- NEW DEFAULTS FOR SEA ---
                "block_variance_thresh": 40.0,
                "width_filter_divisor": 1.5 
            },
            "land": {
                "edges_threshold": 90,          
                "hough_threshold": 150,         
                "hough_max_line_gap": 40,       
                "horizon_angle_tol_deg": 15.0,  
                "hue_thresh": 40.0,             
                "sat_thresh": 40.0,
                "val_thresh": 40.0,
                # --- NEW DEFAULTS FOR LAND ---
                "block_variance_thresh": 25.0,  # Lower variance threshold for land
                "width_filter_divisor": 1.2     # Stricter width filtering
            }
        }

        if mode not in presets:
            raise ValueError("Mode must be exactly 'sea' or 'land'")
        
        p = presets[mode]

        self.edges_threshold = edges_threshold if edges_threshold is not None else p["edges_threshold"]
        self.hough_threshold = hough_threshold if hough_threshold is not None else p["hough_threshold"]
        self.hough_max_line_gap = hough_max_line_gap if hough_max_line_gap is not None else p["hough_max_line_gap"]
        self.horizon_angle_tol_deg = horizon_angle_tol_deg if horizon_angle_tol_deg is not None else p["horizon_angle_tol_deg"]
        self.hue_thresh = hue_thresh if hue_thresh is not None else p["hue_thresh"]
        self.sat_thresh = sat_thresh if sat_thresh is not None else p["sat_thresh"]
        self.val_thresh = val_thresh if val_thresh is not None else p["val_thresh"]
        
        # --- ASSIGN NEW VARIABLES ---
        self.block_variance_thresh = block_variance_thresh if block_variance_thresh is not None else p["block_variance_thresh"]
        self.width_filter_divisor = width_filter_divisor if width_filter_divisor is not None else p["width_filter_divisor"]

        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.grid_size = grid_size
        self.min_contiguous_blocks = min_contiguous_blocks
        self.min_crop_px = min_crop_px

        self.img_width = None
        self.img_height = None
        
        if image is not None:
            self.img_height, self.img_width = image.shape[:2]
            if hough_min_line_length is None:
                self.hough_min_line_length = int(self.img_width * 0.4)
            else:
                self.hough_min_line_length = hough_min_line_length
        else:
            self.hough_min_line_length = hough_min_line_length if hough_min_line_length is not None else 100

class ROIDetector: 
    def __init__(self, config: ROIConfig = None):
        self.config = config if config is not None else ROIConfig()

    #Step 1: Horizon mask: np.array --> y:axis int
    # y = -1 -> no horizon line
    def detect_horizon(self, image: np.ndarray) -> tuple[int, np.array, np.array]:
        #Returns: horizon level:int, blur: np.array, edges: np.array
        y = -1 
        
        # Get height and width of image
        h, w = image.shape[:2] 

        blurred = cv2.GaussianBlur(image, (1,9), 0)
        
        kernel_h = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
        ], dtype=np.float32)

        kernel_v = np.array([
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
        ], dtype=np.float32)

        sobel_horizontal = cv2.filter2D(blurred, cv2.CV_64F, kernel_h)
        sobel_vertical = cv2.filter2D(blurred, cv2.CV_64F, kernel_v)
        #Get L^2 through pythogoras theorem
        sobel_combined = cv2.magnitude(sobel_horizontal, sobel_vertical)
        # convert back to 8-bit visual images (0-255)
        edges = cv2.convertScaleAbs(sobel_combined)
        _, edges = cv2.threshold(edges, self.config.edges_threshold, 255, cv2.THRESH_BINARY)
        
        horizon_lines = cv2.HoughLinesP(
            cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY), 
            rho=self.config.hough_rho, 
            theta=self.config.hough_theta, 
            threshold=self.config.hough_threshold, 
            minLineLength=self.config.hough_min_line_length, 
            maxLineGap=self.config.hough_max_line_gap
        )

        if horizon_lines is not None and len(horizon_lines) > 0:
            flat_lines = [line[0] for line in horizon_lines]
            longest_line = max(flat_lines, key=lambda p: (p[2] - p[0])**2 + (p[3] - p[1])**2)
            x1, y1, x2, y2 = longest_line
            # Midpoint Y of the longest line — avoids the old x=0 extrapolation
            # (c = y1 - m*x1) that went negative when the segment was far from the left edge.
            y = int((y1 + y2) / 2)
 
        return (y, blurred, edges)

    def debug_horizon(self, image: np.ndarray):
        """Visualizer: Plots Original, Canny Edges, and Hough Lines side-by-side."""
        
        blurred = cv2.GaussianBlur(image, (1,9), 0)
        
        kernel_h = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
        ], dtype=np.float32)

        kernel_v = np.array([
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
        ], dtype=np.float32)
        sobel_horizontal = cv2.filter2D(blurred, cv2.CV_64F, kernel_h)
        sobel_vertical = cv2.filter2D(blurred, cv2.CV_64F, kernel_v)
        #Get L^2 through pythogoras theorem
        sobel_combined = cv2.magnitude(sobel_horizontal, sobel_vertical)
        
        # convert back to 8-bit visual images (0-255)
        edges = cv2.convertScaleAbs(sobel_combined)
        _, edges = cv2.threshold(edges, self.config.edges_threshold, 255, cv2.THRESH_BINARY)

        horizon_lines = cv2.HoughLinesP(
            cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY), 
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
 
            # Apply the same angle filter as detect_horizon
            tol = self.config.horizon_angle_tol_deg
            near_horizontal = [
                (x1, y1, x2, y2) for x1, y1, x2, y2 in flat_lines
                if abs(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1) + 1e-6))) <= tol
            ]
            if not near_horizontal:
                near_horizontal = flat_lines
 
            # Use median midpoint Y — same as the fixed detect_horizon
            mid_ys = [(y1 + y2) / 2 for x1, y1, x2, y2 in near_horizontal]
            y_median = int(np.median(mid_ys))
            y_max = y_median
 
            # Draw the horizon line across the full image width
            cv2.line(line_img, (0, y_median), (self.config.img_width, y_median), (0, 0, 255), 2)
 
            # Highlight the longest near-horizontal line in green for reference
            longest_line = max(near_horizontal, key=lambda p: (p[2] - p[0])**2 + (p[3] - p[1])**2)
            x1, y1, x2, y2 = longest_line
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
 
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

    def compute_baseline(self, guassian_rgb: np.ndarray, horizon_y:int) -> np.ndarray:
        #3. Dynamic Baseline -> find the median ** MIGHT CHANGE NEXT TIME
        # If no Horizon, assume all is water. Or object is so close you cant even detect anyways
        # 1. Convert the entire image to HSV
        hsv_image = cv2.cvtColor(guassian_rgb, cv2.COLOR_BGR2HSV)
        
        # 2. Determine where to slice the image
        if horizon_y > 0 and horizon_y < self.config.img_height:
            # start at horizon_Y + 1 
            start_y = horizon_y 
        else:
            start_y = 0 # No horizon, use the whole image

        # 3. Slice the array
        # This grabs everything from start_y down to the bottom, across all X columns
        water_region = hsv_image[start_y:, :]
        
        # 4. Flatten the 2D region into a 1D list of pixels, then find the median
        flat_pixels = water_region.reshape(-1, 3)
        baseline_hsv = np.median(flat_pixels, axis=0)
        
        return baseline_hsv

    def anomaly_detection(self, image: np.ndarray, grid: list, water_baseline: np.ndarray, 
                          blocks_to_check: list = None, custom_thresh: float = None) -> dict:
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        anomalies = {}
        #Scan entire image or blocks of image
        if custom_thresh:
            h_t = s_t = v_t = custom_thresh
        else: 
            h_t = self.config.hue_thresh
            s_t = self.config.sat_thresh
            v_t = self.config.val_thresh

        #Blocks
        if blocks_to_check is None:
            num_rows = len(grid)
            num_cols = len(grid[0])
            blocks_to_check = [(r, c) for r in range(num_rows) for c in range(num_cols)]

        
        num_rows = len(grid)
        num_cols = len(grid[0])
        

        #Only check blocks we are interested in
            
        for r, c in blocks_to_check:
            x, y, w, h = grid[r][c]
            block_pixels = hsv_image[y:y+h, x:x+w]
            block_hsv = np.mean(block_pixels, axis=(0, 1))

            if block_pixels.size > 0:
                block_hsv = np.mean(block_pixels, axis=(0, 1))
                delta = np.abs(block_hsv - water_baseline)
                
                h_diff = min(delta[0], 180 - delta[0]) 
                s_diff = delta[1]
                v_diff = delta[2]

                block_variance = np.var(block_pixels[:, :, 2]) 
                delta = np.abs(block_hsv - water_baseline)
                h_diff = min(delta[0], 180 - delta[0])
                
                if ((h_diff > h_t) or ...) and (block_variance > self.config.block_variance_thresh):
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

    def spatial_validation(self, anomalies_dict: dict,factor:int) -> list[list[tuple[int, int]]]:
        visited = set()
        valid_objects = []
        
        # 8-way directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        color_tolerance = 40.0 # How much the Hue can change before we consider it a different object

        #BFS for neighbours
        for start_coord, start_hsv in anomalies_dict.items():
            if start_coord in visited: #Prune those visitedd
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
            
            # Min number of grids to group squares into targets
            if len(current_object_coords) < self.config.min_contiguous_blocks:
                continue
                
            columns = [c for r, c in current_object_coords]
            object_width = max(columns) - min(columns) + 1
            
            # Delete wide objects (the trees) (half of grid size)
            if object_width >= (self.config.grid_size * factor) / self.config.width_filter_divisor:
                continue
                
            valid_objects.append(current_object_coords)

        return valid_objects
    
    def simple_spatial_validation(self, anomalies_dict: dict,factor:int) -> list[list[tuple[int, int]]]:
        visited = set()
        valid_objects = []
        
        # 8-way directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        color_tolerance = 30.0 # How much the Hue can change before we consider it a different object

        #BFS for neighbours
        for start_coord, start_hsv in anomalies_dict.items():
            if start_coord in visited: #Prune those visitedd
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
            
            # Min number of grids to group squares into targets
            if len(current_object_coords) < self.config.min_contiguous_blocks:
                continue
                
            columns = [c for r, c in current_object_coords]
            object_width = max(columns) - min(columns) + 1
            
            # Delete wide objects (the trees) (half of grid size)
            if object_width >= self.config.grid_size * factor / 1.5 :
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

    def extract_rois(self, image: np.ndarray, grid: list, valid_objects: list, target_size=(64, 64)) -> list:
        """
        Step 6: ROI Crop
        Takes the grouped grid coordinates, calculates exact pixel bounding boxes,
        crops the original image, resizes them for Phase 2, and returns the data.
        64 x 64 since FFT is fastest with  2^ this is the goldilocks zone
        """
        rois = []
        h_img, w_img = image.shape[:2]

        for obj_coords in valid_objects:
            # 1. Find the grid boundaries of the object
            rows = [r for r, c in obj_coords]
            cols = [c for r, c in obj_coords]
            
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            
            # 2. Convert grid coordinates back to exact pixel coordinates
            x1 = grid[min_r][min_c][0]
            y1 = grid[min_r][min_c][1]
            
            x_br, y_br, w_br, h_br = grid[max_r][max_c]
            x2 = x_br + w_br
            y2 = y_br + h_br
            
            # 3. Add a 5-pixel padding so we don't accidentally slice the edge of the buoy off
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w_img, x2 + padding)
            y2 = min(h_img, y2 + padding)
            
            w = x2 - x1
            h = y2 - y1

            # 4. SLICE THE ARRAY (The actual crop!)
            bgr_crop = image[y1:y2, x1:x2]
            
            # 5. Resize to a standard dimension (e.g., 64x64) so the RCE network doesn't crash
            if bgr_crop.size > 0:
                bgr_resized = cv2.resize(bgr_crop, target_size)

                
                # 6. Package it as a tuple: (cropped_array, (x, y, w, h))
                rois.append((bgr_resized, (x1, y1, w, h)))

        return rois
    
    def debug_rois(self, rois: list, rois_sobel: list):
        """
        Visualizer: Plots a gallery of the extracted ROIs.
        Displays Gaussian and Sobel outputs side-by-side for direct comparison.
        Max 2 targets (4 columns) per row.
        """
        num_rois = len(rois)
        
        if num_rois == 0:
            print("Debug ROIs: No valid targets found to display.")
            return
        
        # 1. Calculate grid dimensions (2 pairs per row = 4 columns max)
        pairs_per_row = 2
        # If there is only 1 ROI, make it 2 columns. Otherwise, 4 columns.
        cols = min(num_rois, pairs_per_row) * 2 
        rows = (num_rois + pairs_per_row - 1) // pairs_per_row 

        # 2. Create the dynamic subplots
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        
        # 3. Flatten the axes array for easy iteration
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes] # Handle edge case of 1 row, 1 col (though minimum is 2 cols here)

        # 4. Loop through the ROIs and place them in the grid pairs
        for i in range(num_rois):
            crop_gauss, bbox = rois[i]
            crop_sobel, _ = rois_sobel[i] # The bounding box is identical for both
            
            # NOTE: If your ROIs are ALREADY in RGB format from earlier steps, 
            # remove these cvtColor lines so you don't recreate the "orange ocean" bug!
            rgb_gauss = cv2.cvtColor(crop_gauss, cv2.COLOR_BGR2RGB)
            rgb_sobel = cv2.cvtColor(crop_sobel, cv2.COLOR_BGR2RGB)
            
            x, y, w, h = bbox
            
            # Calculate where this pair goes in the flattened axis array
            gauss_idx = i * 2
            sobel_idx = i * 2 + 1
            
            # Plot Gaussian
            axes[gauss_idx].imshow(rgb_gauss)
            axes[gauss_idx].set_title(f"Target {i+1} (Gaussian)\nBBox: (x:{x}, y:{y})")
            axes[gauss_idx].axis('off')
            
            # Plot Sobel
            axes[sobel_idx].imshow(rgb_sobel)
            axes[sobel_idx].set_title(f"Target {i+1} (Sobel)\nShape: {w}x{h}")
            axes[sobel_idx].axis('off')

        # 5. Hide any empty grid slots at the end of the last row
        for j in range(num_rois * 2, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f'Phase 1 Final Payload: {num_rois} ROIs Extracted', fontsize=16)
        plt.tight_layout()
        plt.show()
        
    def process(self, image: np.ndarray) -> tuple[list[np.array],list[np.array],list[np.array]]:
        #Output 

        #1. Detect Horizon
        # 1. Detect Horizon
        horizon_y,zero_order,first_order = self.detect_horizon(image)
        
        # 2. Generate Dual Grids
        factor = 2 
        coarse_grid = self.tessellate_grid(image, size=self.config.grid_size)
        fine_grid = self.tessellate_grid(image, size=self.config.grid_size * factor)

        # 3. Dynamic Baseline
        baseline = self.compute_baseline(zero_order, horizon_y)

        # 4. Anomaly Detection (Coarse -> Fine)
        coarse_anomalies = self.anomaly_detection(image, coarse_grid, baseline, custom_thresh=15.0)
        
        fine_blocks_to_check = []
        for r, c in coarse_anomalies.keys():
            for i in range(factor):
                for j in range(factor):
                    fine_blocks_to_check.append((r * factor + i, c * factor + j))

        fine_anomalies = self.anomaly_detection(
            image, fine_grid, baseline, 
            blocks_to_check=fine_blocks_to_check
        )

        valid_objects = self.spatial_validation(fine_anomalies,factor) 

        # 6. ROI Crop
        rois_gauss = self.extract_rois(zero_order, fine_grid, valid_objects)
        rois_sobel = self.extract_rois(first_order, fine_grid, valid_objects)
        #Zero order is guassian blurred RGB, first order is sobel RGB

        final_targets = []
        for i in range(len(rois_gauss)):
            gauss_crop, bbox = rois_gauss[i]
            sobel_crop, _ = rois_sobel[i] # BBox is identical, so we can ignore the second one
            
            target_data = {
                "id": i,
                "bbox": bbox,               # (x, y, w, h) from the original image
                "gaussian_bgr": gauss_crop, # The 3 zero-order matrices BGR
                "sobel_bgr": sobel_crop     # The 3 first-order matrices BGR
            }
            final_targets.append(target_data)

        return final_targets

    def simple_process(self, frame):
        """
        Bypasses the anomaly detector entirely.
        Slices the entire image into a grid of 64x64 crops and sends 
        EVERY block to the RCE Neural Network for evaluation.
        """
        targets = []
        
        # 1. Get dimensions
        h_img, w_img = frame.shape[:2]
        
        # 2. Compute the Phase 1 matrices for the entire image at once (Massive Optimization)
        gaussian_bgr = cv2.GaussianBlur(frame, (1, 9), 0)
        
        kernel_h = np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]], dtype=np.float32)
        kernel_v = np.array([[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]], dtype=np.float32)
        
        sobel_h = cv2.filter2D(gaussian_bgr, cv2.CV_64F, kernel_h)
        sobel_v = cv2.filter2D(gaussian_bgr, cv2.CV_64F, kernel_v)
        sobel_combined = cv2.magnitude(sobel_h, sobel_v)
        sobel_bgr = cv2.convertScaleAbs(sobel_combined)
        
        # 3. Slice the image into 64x64 blocks
        crop_size = 64
        target_id = 0
        
        # How many pixels the window shifts before taking the next crop.
        # 64 = No overlap (Fastest). 32 = 50% overlap (More accurate, but 4x slower)
        step_size = 64 
        
        for y in range(0, h_img - crop_size + 1, step_size):
            for x in range(0, w_img - crop_size + 1, step_size):
                
                # Extract the exact 64x64 crops from the pre-calculated matrices
                gauss_crop = gaussian_bgr[y:y+crop_size, x:x+crop_size]
                sobel_crop = sobel_bgr[y:y+crop_size, x:x+crop_size]
                
                # Package it perfectly for Phase 2
                target = {
                    "id": target_id,
                    "bbox": (x, y, crop_size, crop_size),
                    "gaussian_bgr": gauss_crop,
                    "sobel_bgr": sobel_crop
                }
                targets.append(target)
                target_id += 1
                
        # Optional: Print out how many blocks it generated
        # print(f"[*] Sliced image into {len(targets)} 64x64 blocks for RCE evaluation.")
        
        return targets

    def debug_process(self, image: np.ndarray):
        # 3. Call the methods directly on the instance
        print("\n--- Starting Phase 1 Debug Process ---")
        
        # 1. Detect Horizon
        horizon_y,zero_order,first_order = self.detect_horizon(image)
        print(f"Horizon Y-Coordinate: {horizon_y}")
        self.debug_horizon(image)

        # 2. Generate Dual Grids
        factor = 2 
        coarse_grid = self.tessellate_grid(image, size=self.config.grid_size)
        fine_grid = self.tessellate_grid(image, size=self.config.grid_size * factor)

        # 3. Dynamic Baseline
        baseline = self.compute_baseline(zero_order, horizon_y)
        print(f"Water Baseline Median HSV: {baseline}")

        # 4. Anomaly Detection (Coarse -> Fine)
        coarse_anomalies = self.anomaly_detection(image, coarse_grid, baseline, custom_thresh=15.0)
        self.debug_anomalies(image, coarse_grid, coarse_anomalies)

        fine_blocks_to_check = []
        for r, c in coarse_anomalies.keys():
            for i in range(factor):
                for j in range(factor):
                    fine_blocks_to_check.append((r * factor + i, c * factor + j))

        fine_anomalies = self.anomaly_detection(
            image, fine_grid, baseline, 
            blocks_to_check=fine_blocks_to_check, 
            custom_thresh=35.0
        )
        self.debug_anomalies(image, fine_grid, fine_anomalies)

        # 5. Spatial Validation
        valid_objects = self.spatial_validation(fine_anomalies,factor)
        self.debug_spatial_validation(image, fine_grid, valid_objects)

        # 6. ROI Crop
        rois = self.extract_rois(zero_order, fine_grid, valid_objects)
        rois_sobel = self.extract_rois(first_order, fine_grid, valid_objects)
        
        # New call passing both lists:
        self.debug_rois(rois, rois_sobel)
        
        return (rois, zero_order, first_order)
    
    def debug_simple_process(self, image: np.ndarray):
        # 3. Call the methods directly on the instance
        print("\n--- Starting Phase 1 Debug Process ---")
        
        # 1. Detect Horizon
        _,zero_order,first_order = self.detect_horizon(image)
        print(f"Horizon Y-Coordinate: {0}")
        self.debug_horizon(image)

        # 2. Generate Dual Grids
        factor = 2 
        coarse_grid = self.tessellate_grid(image, size=self.config.grid_size)
        fine_grid = self.tessellate_grid(image, size=self.config.grid_size * factor)

        # 3. Dynamic Baseline
        # baseline = self.compute_baseline(zero_order, 0)
        print(f"Water Baseline Median HSV: {0,0,0}")

        # 4. Anomaly Detection (Coarse -> Fine)
        coarse_anomalies = self.anomaly_detection(image, coarse_grid, (0,0,0), custom_thresh=15.0)
        self.debug_anomalies(image, coarse_grid, coarse_anomalies)

        fine_blocks_to_check = []
        for r, c in coarse_anomalies.keys():
            for i in range(factor):
                for j in range(factor):
                    fine_blocks_to_check.append((r * factor + i, c * factor + j))

        fine_anomalies = self.anomaly_detection(
            image, fine_grid, (0,0,0), 
            blocks_to_check=fine_blocks_to_check, 
            custom_thresh=35.0
        )
        self.debug_anomalies(image, fine_grid, fine_anomalies)

        # 5. Spatial Validation
        valid_objects = self.simple_spatial_validation(fine_anomalies,factor)
        self.debug_spatial_validation(image, fine_grid, valid_objects)

        # 6. ROI Crop
        rois = self.extract_rois(zero_order, fine_grid, valid_objects)
        rois_sobel = self.extract_rois(first_order, fine_grid, valid_objects)
        
        # New call passing both lists:
        self.debug_rois(rois, rois_sobel)
        
        return (rois, zero_order, first_order)

    def run_live_feed(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return

        print("Starting Live Feed... Press 'q' to quit.")

        prev_time = time.perf_counter()
        fps = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            start_process = time.perf_counter()
            
            # FIXED: Calls its own process method
            rois = self.process(frame)
            
            end_process = time.perf_counter()
            
            process_time_ms = (end_process - start_process) * 1000
            
            curr_time = time.perf_counter()
            fps_instant = 1.0 / (curr_time - prev_time)
            fps = fps * 0.9 + fps_instant * 0.1 
            prev_time = curr_time

            display_frame = frame.copy()
            
            for i, (crop, (x, y, w, h)) in enumerate(rois):
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"Target {i+1}"
                cv2.putText(display_frame, label, (x, max(0, y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (350, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
            
            cv2.putText(display_frame, "VISION SYSTEM DASHBOARD", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            fps_color = (0, 255, 0) if fps > 20 else (0, 0, 255)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (15, 65), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
                        
            cv2.putText(display_frame, f"Process Time: {process_time_ms:.1f} ms", (15, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            target_color = (0, 255, 0) if len(rois) > 0 else (200, 200, 200)
            cv2.putText(display_frame, f"Active Targets: {len(rois)}", (15, 125), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)

            cv2.imshow("Phase 1 - Real-Time Output", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Create an empty config (it will auto-calculate sizes based on the first frame it sees)
    config = ROIConfig()
    roi_detector = ROIDetector(config)
    
    # Run the live feed! 
    # Use 0 for your webcam, or put a path to an mp4 file like "TestVideo.mp4"
    #roi_detector.run_live_feed(video_source=0)
    
    image_path = os.path.join(os.getcwd(), "test_4_red_cross.jpg")
    img = cv2.imread(image_path)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    if img is None:
        print("ERROR! NO IMG FOUND!")
        exit()
    config = ROIConfig(img)
    roi_detector = ROIDetector(config)

    roi_detector.debug_simple_process(img)
    targets = roi_detector.simple_process(img)

