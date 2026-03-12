from ROIDetector import *

import cv2
import time
import os
import numpy as np

# ... (Keep your ROIConfig and ROIDetector classes exactly as they are) ...

def run_live_feed(roi_detector: ROIDetector, video_source=0):
    """
    Runs the Phase 1 pipeline on a live video feed or MP4 file.
    Draws a real-time HUD (Heads-Up Display) with metrics and bounding boxes.
    """
    # Initialize the video capture (0 is usually your laptop webcam)
    # To use a video file, replace 0 with the path: 'robotx_test_footage.mp4'
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    print("Starting Live Feed... Press 'q' to quit.")

    # Variables for smoothing out the FPS calculation
    prev_time = time.perf_counter()
    fps = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
            
        # Optional: Resize massive 4K frames down to 1080p or 720p for processing speed
        # frame = cv2.resize(frame, (1280, 720)) 

        # --- 1. START STOPWATCH & PROCESS ---
        start_process = time.perf_counter()
        
        # Run your entire silent Phase 1 pipeline on the current frame
        rois = roi_detector.process(frame)
        
        end_process = time.perf_counter()
        
        # --- 2. CALCULATE METRICS ---
        process_time_ms = (end_process - start_process) * 1000
        
        curr_time = time.perf_counter()
        fps_instant = 1.0 / (curr_time - prev_time)
        # Use an Exponential Moving Average to stop the FPS text from flickering wildly
        fps = fps * 0.9 + fps_instant * 0.1 
        prev_time = curr_time

        # --- 3. DRAW VISUALS (Bounding Boxes) ---
        display_frame = frame.copy()
        
        # Extract the bounding box coordinates from the Phase 1 payload and draw them
        for i, (crop, (x, y, w, h)) in enumerate(rois):
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Label the target just above the bounding box
            label = f"Target {i+1}"
            cv2.putText(display_frame, label, (x, max(0, y - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- 4. DRAW DASHBOARD (HUD) ---
        # Create a dark, semi-transparent background box for the text
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, 140), (0, 0, 0), -1)
        # Blend the overlay with the original frame (40% opacity)
        cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
        
        # Write the Dashboard Metrics
        cv2.putText(display_frame, "VISION SYSTEM DASHBOARD", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Color the FPS green if it's healthy (>20), red if it's lagging
        fps_color = (0, 255, 0) if fps > 20 else (0, 0, 255)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (15, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
                    
        cv2.putText(display_frame, f"Process Time: {process_time_ms:.1f} ms", (15, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Highlight target count in green if targets are found
        target_color = (0, 255, 0) if len(rois) > 0 else (200, 200, 200)
        cv2.putText(display_frame, f"Active Targets: {len(rois)}", (15, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)

        # --- 5. RENDER FRAME ---
        cv2.imshow("Phase 1 - Real-Time Output", display_frame)

        # Graceful exit: Press 'q' to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Clean up hardware resources
    cap.release()
    cv2.destroyAllWindows()

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Create an empty config (it will auto-calculate sizes based on the first frame it sees)
    config = ROIConfig()
    roi_detector = ROIDetector(config)
    
    # Run the live feed! 
    # Use 0 for your webcam, or put a path to an mp4 file like "TestVideo.mp4"
    run_live_feed(roi_detector, video_source=0)