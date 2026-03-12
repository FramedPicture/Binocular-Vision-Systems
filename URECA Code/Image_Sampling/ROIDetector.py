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

Coarse to fine 
Loose grid + fine grid -> run grid tesslation and Anomly detection twice
Anomly detection -> Lower grid size, lower HSV diff threshold (less sensitive)

Input : raw left-camera BGR image (np.ndarray, shape H x W x 3 HSV)
Output: list of (crop_bgr, (x, y, w, h)) tuples
"""

import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import time

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
        grid_size=8, #For grid tesslation
        hue_thresh=20.0, #For Anomaly detection (was 40)
        sat_thresh=20.0,
        val_thresh=20.0,
        min_contiguous_blocks=1, #For Neighbour mnatching
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

    #Step 1: Horizon mask: np.array --> y:axis int
    # y = -1 -> no horizon line
    def detect_horizon(self, image: np.ndarray) -> int:
        y = -1 
        
        # 1. Grab the exact height and width of the current live frame!
        h, w = image.shape[:2] 

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Added 'self.' to the config variables
        edges = cv2.Canny(blur, self.config.canny_low, self.config.canny_high)

        horizon_lines = cv2.HoughLinesP(
            edges, 
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

            if x2 != x1:
                m = (y2 - y1) / (x2 - x1)  # Slope (m)
                c = y1 - (m * x1)          # Intercept (c)
            
                left_y = int(c)
                # 3. Use the dynamic 'w' we grabbed from the frame, NOT the config!
                right_y = int(m * w + c)   
                y = min(left_y, right_y)
            else: 
                y = min(y1, y2)

        return y

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
        # If no Horizon, assume all is water. Or object is so close you cant even detect anyways
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
                
                if ((h_diff > h_t) or (s_diff > s_t) or (v_diff > v_t)) and (block_variance > 40.0):
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
            if object_width >= self.config.grid_size * factor /2 :
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
            
            # Optional: If you saved self.edge_map in Step 1, crop it here too!
            # edge_crop = self.edge_map[y1:y2, x1:x2]
            
            # 5. Resize to a standard dimension (e.g., 64x64) so the RCE network doesn't crash
            if bgr_crop.size > 0:
                bgr_resized = cv2.resize(bgr_crop, target_size)
                # edge_resized = cv2.resize(edge_crop, target_size)
                
                # 6. Package it as a tuple: (cropped_array, (x, y, w, h))
                rois.append((bgr_resized, (x1, y1, w, h)))
                
                # If passing the edge map too, it would look like this:
                # rois.append((bgr_resized, edge_resized, (x1, y1, w, h)))

        return rois
    
    def debug_rois(self, rois: list):
        """
        Visualizer: Plots a gallery of the final extracted and resized ROIs.
        Displays a maximum of 5 ROIs per row and wraps to a new row.
        """
        num_rois = len(rois)
        
        if num_rois == 0:
            print("Debug ROIs: No valid targets found to display.")
            return

        # 1. Calculate the grid dimensions (max 5 columns)
        max_cols = 5
        cols = min(num_rois, max_cols)
        rows = (num_rois + max_cols - 1) // max_cols # Math trick to round up division

        # 2. Create the dynamic subplots
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        
        # 3. Flatten the axes array so we can loop through it easily 
        # (Matplotlib makes a 2D array if rows > 1, or a 1D array if rows == 1)
        if type(axes) is np.ndarray:
            axes = axes.flatten()
        else:
            axes = [axes] # Handle the edge case where there is exactly 1 ROI

        # 4. Loop through the grid slots
        for i, ax in enumerate(axes):
            if i < num_rois:
                # We have an ROI to display in this slot!
                crop_bgr, bbox = rois[i]
                
                # Convert BGR to RGB for Matplotlib
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                x, y, w, h = bbox
                
                ax.imshow(crop_rgb)
                ax.set_title(f"Target {i+1}\nBBox: (x:{x}, y:{y}, w:{w}, h:{h})\nShape: {crop_rgb.shape}")
                ax.axis('off')
            else:
                # We ran out of ROIs, but there are empty grid slots left. Hide them.
                ax.axis('off')

        plt.suptitle(f'Phase 1 Final Payload: {num_rois} ROIs Ready for Phase 2', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def process(self, image: np.ndarray):
        #Run the whole ROIDetection

        #1. Detect Horizon
        # 1. Detect Horizon
        horizon_y = self.detect_horizon(image)
        
        # 2. Generate Dual Grids
        factor = 2 
        coarse_grid = self.tessellate_grid(image, size=self.config.grid_size)
        fine_grid = self.tessellate_grid(image, size=self.config.grid_size * factor)

        # 3. Dynamic Baseline
        baseline = self.compute_baseline(image, horizon_y)

        # 4. Anomaly Detection (Coarse -> Fine)
        coarse_anomalies = self.anomaly_detection(image, coarse_grid, baseline, custom_thresh=15.0)
        
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

        # 5. Spatial Validation (Assuming you passed factor in your updated version)
        valid_objects = self.spatial_validation(fine_anomalies,factor) 

        # 6. ROI Crop
        rois = self.extract_rois(image, fine_grid, valid_objects)
        
        return rois

    def debug_process(self, image: np.ndarray):
        # 3. Call the methods directly on the instance
        print("\n--- Starting Phase 1 Debug Process ---")
        
        # 1. Detect Horizon
        horizon_y = self.detect_horizon(image)
        print(f"Horizon Y-Coordinate: {horizon_y}")
        self.debug_horizon(image)

        # 2. Generate Dual Grids
        factor = 2 
        coarse_grid = self.tessellate_grid(image, size=self.config.grid_size)
        fine_grid = self.tessellate_grid(image, size=self.config.grid_size * factor)

        # 3. Dynamic Baseline
        baseline = self.compute_baseline(image, horizon_y)
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
        rois = self.extract_rois(image, fine_grid, valid_objects)
        self.debug_rois(rois)

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
    
    image_path = os.path.join(os.getcwd(), "Test2.jpg")
    img = cv2.imread(image_path)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    if img is None:
        print("ERROR! NO IMG FOUND!")
        exit()
    config = ROIConfig(img)
    roi_detector = ROIDetector(config)

    roi_detector.debug_process(img)
