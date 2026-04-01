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
Feature matrix Fj (24 x 1 int Vector)
Guassaian R G B (3) x mean_energy, var_energy, var_x_energy, var_y_energy(4) = 12 x 1  scalar 
Sobel R G B (3) x mean_energy, var_energy, var_x_energy, var_y_energy(4) = 12 x 1 scalar 
Total = 1D Array of 24 x 1 floating-point numbers
======================
Active Pipeline:
Step 1. Take a live ROI crop and compute its 24-value Feature Vector Fj
Step 2. Match with database using Possibility Function
Step 3. Select best match using Soft-Max Function
Step 4. Threshold confidence level (e.g., > 0.5)
Step 5. Output [Label, Confidence, Bounding Box]
'''

import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import time

import os
import cv2
import numpy as np
import json
from typing import List, Tuple, TypedDict

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
# Configuration & Feature Extractor
# ==========================================
class RCEConfig:
    def __init__(self):
        self.p_min = 0.5 
        self.epsilon = 1e-8 
        self.min_sigma = 70
        self.db_filepath = "rce_database.txt" # The persistence file

class FeatureExtractor:
    def __init__(self, config: RCEConfig):
        self.config = config

    def compute_channel_stats(self, channel: np.ndarray) -> List[float]:
        I = channel.astype(np.float64)
        h, w = I.shape
        
        I_a = np.mean(I)
        sigma_I = np.std(I)
        
        S = np.sum(I) + self.config.epsilon
        v_indices, u_indices = np.indices((h, w))
        
        u_c = np.sum(I * u_indices) / S
        v_c = np.sum(I * v_indices) / S
        
        sigma_u = np.sqrt(np.sum(I * (u_indices - u_c)**2) / S)
        sigma_v = np.sqrt(np.sum(I * (v_indices - v_c)**2) / S)
        
        return [I_a, sigma_I, sigma_u, sigma_v]

    def extract_feature_vector(self, target: TargetPayload) -> np.ndarray:
        features = []
        b0, g0, r0 = cv2.split(target["gaussian_bgr"])
        b1, g1, r1 = cv2.split(target["sobel_bgr"])
        
        matrices = [r0, g0, b0, r1, g1, b1]
        for matrix in matrices:
            features.extend(self.compute_channel_stats(matrix))
            
        return np.array(features, dtype=np.float64).reshape(24, 1)

# ==========================================
# Training Pipeline (Appends to .txt)
# ==========================================
class RCECognizer:
    def __init__(self, config: RCEConfig):
        self.config = config
        self.extractor = FeatureExtractor(config)

    def train(self, label: str, training_targets: List[TargetPayload]):
        print(f"Training RCE node for '{label}'...")
        
        feature_vectors = [self.extractor.extract_feature_vector(t) for t in training_targets]
        fv_array = np.array(feature_vectors)
        
        mean_vector = np.mean(fv_array, axis=0) 
        
        distances = []
        for fv in feature_vectors:
            dist_sq = np.dot((fv - mean_vector).T, (fv - mean_vector))[0, 0]
            distances.append(dist_sq)
            
        std_dev = np.sqrt(np.mean(distances))
        if std_dev < self.config.min_sigma:
            std_dev = self.config.min_sigma
            
        # Serialize and APPEND to the text file
        node = {
            "label": label,
            "mean_vector": mean_vector.flatten().tolist(), # Numpy arrays can't be JSON serialized directly
            "std_dev": float(std_dev)
        }
        
        with open(self.config.db_filepath, 'a') as f:
            f.write(json.dumps(node) + "\n")
            
        print(f"Successfully saved '{label}' to {self.config.db_filepath}")

# ==========================================
# Active Pipeline (Reads from .txt)
# ==========================================
class RCERecognizer:
    def __init__(self, config: RCEConfig):
        self.config = config
        self.extractor = FeatureExtractor(config)
        self.database = self._load_database()

    def _load_database(self) -> List[DatabaseNode]:
        """Loads and reconstructs the NumPy arrays from the text file."""
        db = []
        if not os.path.exists(self.config.db_filepath):
            print(f"Warning: {self.config.db_filepath} not found. Database is empty.")
            return db

        with open(self.config.db_filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Reconstruct the 24x1 numpy array
                    data["mean_vector"] = np.array(data["mean_vector"], dtype=np.float64).reshape(24, 1)
                    db.append(data)
                    
        print(f"Loaded {len(db)} trained nodes from {self.config.db_filepath}")
        return db

    def recognize(self, live_targets: List[TargetPayload]) -> List[list]:
        """Evaluates live ROIs and outputs [Bounding Box, Label, Confidence]"""
        results = []
        if not self.database:
            return results
        for target in live_targets:
            live_fv = self.extractor.extract_feature_vector(target)
            
            best_score = 0.0
            best_label = "Unknown"
            
            for node in self.database:
                mean_vec = node["mean_vector"]
                sigma = node["std_dev"]
                
                dist_sq = np.dot((live_fv - mean_vec).T, (live_fv - mean_vec))[0, 0]

                print(f"[*] Testing {node['label']} | Dist Sq: {dist_sq:.2f} | Sigma: {sigma:.4f}")

                score = np.exp(-dist_sq / (2 * (sigma ** 2)))
                
                if score > best_score:
                    best_score = score
                    best_label = node["label"]
            
            if best_score >= self.config.p_min:
                final_label = best_label
            else:
                final_label = "Background"
                
            # Output format: [Bounding Box, Label, Confidence]
            results.append([target["bbox"], final_label, best_score])
            
        return results