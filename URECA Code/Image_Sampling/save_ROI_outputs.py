import cv2
import os
import numpy as np

# Import your existing classes from the file you provided
# Make sure your original file is named ROIDetector.py
from ROIDetector import ROIConfig, ROIDetector

def save_pipeline_outputs(image_path, output_dir="output"):
    """
    Runs the Phase 1 ROI detection pipeline and saves the visual 
    steps to a specified folder, overwriting previous files.
    """
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}. Check the file path.")
        return

    print(f"Processing '{image_path}'...")

    # Save the original image
    cv2.imwrite(os.path.join(output_dir, "01_original.jpg"), img)

    # Initialize your config and detector
    config = ROIConfig(img, mode="sea")
    detector = ROIDetector(config)

    # --- STEP 1: Detect Horizon & Get Filters ---
    horizon_y, blurred, edges = detector.detect_horizon(img)
    
    cv2.imwrite(os.path.join(output_dir, "02_gaussian_blur.jpg"), blurred)
    cv2.imwrite(os.path.join(output_dir, "03_sobel_edges.jpg"), edges)

    # Draw horizon line on a copy for visualization
    horizon_img = img.copy()
    if horizon_y > 0:
        cv2.line(horizon_img, (0, horizon_y), (config.img_width, horizon_y), (0, 0, 255), 2)
        cv2.putText(horizon_img, f"Horizon Y: {horizon_y}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "04_horizon_line.jpg"), horizon_img)

    # --- STEP 2: Generate Dual Grids ---
    factor = 2 
    coarse_grid = detector.tessellate_grid(img, size=config.grid_size)
    fine_grid = detector.tessellate_grid(img, size=config.grid_size * factor)

    # --- STEP 3: Dynamic Baseline ---
    baseline = detector.compute_baseline(blurred, horizon_y)

    # --- STEP 4: Anomaly Detection (Coarse -> Fine) ---
    coarse_anomalies = detector.anomaly_detection(img, coarse_grid, baseline, custom_thresh=15.0)

    fine_blocks_to_check = []
    for r, c in coarse_anomalies.keys():
        for i in range(factor):
            for j in range(factor):
                fine_blocks_to_check.append((r * factor + i, c * factor + j))

    fine_anomalies = detector.anomaly_detection(
        img, fine_grid, baseline, 
        blocks_to_check=fine_blocks_to_check, 
        custom_thresh=35.0
    )

    # Visualizer: Draw the fine grid and highlight anomalies in RED
    anom_img = img.copy()
    # Draw faint grid
    for row in fine_grid:
        for x, y, w, h in row:
            cv2.rectangle(anom_img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    
    # Draw flagged anomalies
    for r, c in fine_anomalies.keys():
        x, y, w, h = fine_grid[r][c]
        cv2.rectangle(anom_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    cv2.imwrite(os.path.join(output_dir, "05_anomaly_detection.jpg"), anom_img)

    # --- STEP 5: Spatial Validation ---
    valid_objects = detector.spatial_validation(fine_anomalies, factor)
    
    # Visualizer: Draw final valid bounding boxes in GREEN
    spatial_img = img.copy()
    for i, obj_coords in enumerate(valid_objects):
        rows = [r for r, c in obj_coords]
        cols = [c for r, c in obj_coords]
        
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        x1 = fine_grid[min_r][min_c][0]
        y1 = fine_grid[min_r][min_c][1]
        x_br, y_br, w_br, h_br = fine_grid[max_r][max_c]
        x2 = x_br + w_br
        y2 = y_br + h_br
        
        cv2.rectangle(spatial_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(spatial_img, f"Target {i+1}", (x1, max(0, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_dir, "06_spatial_validation.jpg"), spatial_img)

    # --- STEP 6: ROI Crop ---
    rois_gauss = detector.extract_rois(blurred, fine_grid, valid_objects)
    
    # Save the individual 64x64 extracted ROIs
    for i, (gauss_crop, bbox) in enumerate(rois_gauss):
        roi_filename = os.path.join(output_dir, f"07_ROI_Target_{i+1}.jpg")
        cv2.imwrite(roi_filename, gauss_crop)

    print(f"Success! {len(valid_objects)} targets found. Output images saved to the '{output_dir}' folder.")

if __name__ == "__main__":
    # Change "Test1.jpg" to whatever your test image file is named
    TARGET_IMAGE = "Test1.jpg" 
    
    save_pipeline_outputs(TARGET_IMAGE)