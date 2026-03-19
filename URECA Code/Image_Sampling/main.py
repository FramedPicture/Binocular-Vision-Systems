import cv2
import os
import time
from enum import Enum

# Import your custom modules
from RCENeural import *
from ROIDetector import *
from RCETraining import preprocess_and_train

def resize_to_p(frame, size: int = 480):
    """
    Resizes the frame to a size height while maintaining the aspect ratio.
    Prevents massive images from breaking cv2.imshow and speeds up processing.
    """
    height, width = frame.shape[:2]
    
    # If it's already 720p height, just return it
    if height == size:
        return frame
        
    # Calculate scale based on target height of 720
    scale = float(size) / height
    new_width = int(width * scale)
    
    return cv2.resize(frame, (new_width, size))


def train_system(config: RCEConfig, training_paths: dict):
    """
    Handles Phase 2 training. Iterates through folders and appends to rce_database.txt.
    """
    cognizer = RCECognizer(config)
    
    for label, folder_path in training_paths.items():
        if os.path.exists(folder_path):
            preprocess_and_train(folder_path, label, cognizer)
        else:
            print(f"Directory not found: {folder_path} - Skipping '{label}'")


def draw_bounding_boxes(frame, live_results):
    """
    Draws a bounding box, label, and p-value for EVERY detected region,
    regardless of whether it's a confirmed target or background.
    """
    for bbox, label, confidence in live_results:
        x, y, w, h = bbox
        
        # Draw a uniform bounding box (let's use Cyan for high visibility)
        color = (255, 255, 0) # BGR format for Cyan
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Create the text tag: "Label (p=0.XX)"
        text = f"{label} (p={confidence:.2f})"
        
        # Draw the background rectangle for the text
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
        
        # Draw the text in black so it contrasts well against the Cyan background
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Optional: Print to your console terminal as well so you have a running log
        print(f"Detected: {label:15} | p-value: {confidence:.4f} | BBox: {bbox}")
            
    return frame


def run_image_inference(image_path: str, rce_config: RCEConfig, mode: str):
    """
    Runs detection and classification on a single static photo.
    """
    recognizer = RCERecognizer(rce_config)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image at '{image_path}'")
        return
    
    # --- DOWNSIZE APPLIED HERE ---
    frame = resize_to_p(frame,480)
    
    config = ROIConfig(frame,mode)
    roi_detector = ROIDetector(config)
    
    print("Processing image...")
    targets = roi_detector.process(frame)
    if targets:
        live_results = recognizer.recognize(targets)
        frame = draw_bounding_boxes(frame, live_results)
    else:
        print("No ROIs detected in the image.")

    print("Displaying result. Press ANY KEY on the image window to close.")
    cv2.imshow("Maritime Autonomy - Static Image Inference", frame)
    cv2.waitKey(0)  # Wait infinitely until the user presses a key
    cv2.destroyAllWindows()


def run_live_inference(video_source, rce_config: RCEConfig):
    """
    Runs detection and classification on a continuous video feed or webcam.
    """
    # Assuming config needs to be initialized first. Note: You might need to 
    # adjust ROIConfig if it requires a frame on initialization like in run_image_inference.
    roi_detector = ROIDetector() 
    recognizer = RCERecognizer(rce_config)

    # If the user passes "0" as a string, convert it to an int for webcam use
    if str(video_source).isdigit():
        video_source = int(video_source)

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return

    print("Starting video/live inference. Press 'q' on the video window to quit.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            print("End of video stream.")
            break

        # --- DOWNSIZE APPLIED HERE ---
        frame = resize_to_p(frame)

        targets = roi_detector.process(frame)

        if targets:
            live_results = recognizer.recognize(targets)
            frame = draw_bounding_boxes(frame, live_results)

        # FPS Monitor
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Maritime Autonomy - Live Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # 1. EXECUTION SETTINGS (Toggle these to True/False)
    # ---------------------------------------------------------
    RUN_TRAINING        = False
    RUN_PHOTO_TEST      = True
    RUN_VIDEO_TEST      = False
    RUN_LIVE_WEBCAM     = False

    # ---------------------------------------------------------
    # 2. FILE PATHS
    # ---------------------------------------------------------
    TRAINING_PATHS = {
        "Red Cross": "./training/RedCross",
    }
    
    TEST_PHOTO_PATH = "./test_4_red_cross.jpg"
    TEST_VIDEO_PATH = "./test_video.mp4"
    
    # ---------------------------------------------------------
    # 3. RUN PIPELINE
    # ---------------------------------------------------------
    config = RCEConfig()
    
    if RUN_TRAINING:
        print("\n=== Initiating RCE Neural Network Training ===")
        train_system(config, TRAINING_PATHS)
        
    if RUN_PHOTO_TEST:
        print(f"\n=== Running Inference on Photo: {TEST_PHOTO_PATH} ===")
        run_image_inference(TEST_PHOTO_PATH, config, "land")
        
    if RUN_VIDEO_TEST:
        print(f"\n=== Running Inference on Video: {TEST_VIDEO_PATH} ===")
        run_live_inference(TEST_VIDEO_PATH, config)
        
    if RUN_LIVE_WEBCAM:
        print("\n=== Launching Live Webcam Inference ===")
        run_live_inference(0, config)