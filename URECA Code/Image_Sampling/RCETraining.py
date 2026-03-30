'''
Phase 2: RCE Neural network
======================
Input: ROI cooridnate list: 64 x 64 images of RGB guassian and sobel matrix (x,y,R,G,B)
Targets[
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    gaussian_bgr: np.ndarray         # Shape: (64, 64, 3), dtype: uint8
    sobel_bgr: np.ndarray            # Shape: (64, 64, 3), dtype: uint8 ]
Output: Object classification & coordinates on image
        [Bounding Box, Label, Confidence]
======================
Feature matrix Fj (24 x 1 int matrix)
Guassaian R G B (3) x mean_energy, var_energy, var_x_energy, var_y_energy(4) = 12 x 1  scalar 
Sobel R G B (3) x mean_energy, var_energy, var_x_energy, var_y_energy(4) = 12 x 1 scalar 
Total = 1D Array of 24 x 1 floating-point numbers
======================
Training Pipeline:
Step 1. Training data -> Feature matrixes
Step 2. Compute the Mean Vector avg_Fj. Center of hyper-sphere
Step 3. Compute the Standard Deviation var_fj. Radius of hyper sphere.
Step 4. Store pair node (label, avg_Fj, var_fj) in database. 
'''

import os
import cv2
import numpy as np
import json
from typing import List, Tuple, TypedDict
from RCENeural import RCECognizer,RCEConfig

# --- Types ---
class TargetPayload(TypedDict):
    id: int
    bbox: Tuple[int, int, int, int]
    gaussian_bgr: np.ndarray
    sobel_bgr: np.ndarray

class DatabaseNode(TypedDict):
    label: str
    mean_vector: list  # Stored as list in txt, converted to np.array in memory
    std_dev: float

# ==========================================
# Utility: Preprocess & Train from Folder
# ==========================================
def preprocess_and_train(folder_path: str, label: str, cognizer: 'RCECognizer', target_size=(64, 64)):
    """
    Reads all images in a folder, resizes them, applies the required Gaussian/Sobel filters, 
    and feeds them to the Cognizer for training.
    """
    training_payloads = []
    
    # Kernel definitions from your Phase 1
    kernel_h = np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]], dtype=np.float32)
    kernel_v = np.array([[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]], dtype=np.float32)

    for i, filename in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        # 1. Resize to 64x64
        resized_img = cv2.resize(img, target_size)
        
        # 2. Mimic Phase 1 pipeline to get the exact matrices needed
        gaussian_bgr = cv2.GaussianBlur(resized_img, (5,5), 0)
        
        sobel_h = cv2.filter2D(gaussian_bgr, cv2.CV_64F, kernel_h)
        sobel_v = cv2.filter2D(gaussian_bgr, cv2.CV_64F, kernel_v)
        sobel_combined = cv2.magnitude(sobel_h, sobel_v)
        sobel_bgr = cv2.convertScaleAbs(sobel_combined)
        
        # 3. Package into the TargetPayload format
        payload: TargetPayload = {
            "id": i,
            "bbox": (0, 0, target_size[0], target_size[1]), # Dummy bbox for training
            "gaussian_bgr": gaussian_bgr,
            "sobel_bgr": sobel_bgr
        }
        training_payloads.append(payload)
        
    if training_payloads:
        print(f"Successfully processed {len(training_payloads)} images from '{folder_path}'")
        cognizer.train(label, training_payloads)
    else:
        print(f"No valid images found in {folder_path}")


