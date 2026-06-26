import cv2
import os
import glob

def main():
    # 1. Setup folders
    input_folder = "red_buoy"  # Put your raw frames in here
    output_folder = "cropped_dataset"
    
    if not os.path.exists(input_folder):
        print(f"Please create a folder named '{input_folder}' and put your images in it.")
        return
        
    label = input("What are we cropping? (e.g., 'red_buoy'): ").strip()
    
    # Create a specific subfolder for this label (e.g., cropped_dataset/red_buoy/)
    save_dir = os.path.join(output_folder, label)
    os.makedirs(save_dir, exist_ok=True)

    image_files = glob.glob(f"{input_folder}/*.jpg") + glob.glob(f"{input_folder}/*.png")
    if not image_files:
        print(f"No images found in {input_folder}!")
        return

    print("\n--- INSTRUCTIONS ---")
    print("1. Drag a tight box around the target.")
    print("2. Press ENTER or SPACE to confirm.")
    print("3. Press 'c' to skip an image.")
    print("4. Press 'q' to quit.")
    print(f"Outputting 64x64 crops to: {save_dir}\n")

    saved_count = 0

    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is None:
            continue
            
        # Optional: Shrink massive images for display purposes so it fits on your screen
        disp_img = img.copy()
        h, w = disp_img.shape[:2]
        if w > 1280:
            disp_img = cv2.resize(disp_img, (1280, int(h * (1280/w))))

        window_name = f"Cropping: {os.path.basename(img_file)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Open ROI selector
        roi = cv2.selectROI(window_name, disp_img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)

        # ROI returns (x, y, w, h). If w or h is 0, user pressed 'c' to skip.
        x, y, w, h = roi
        if w > 0 and h > 0:
            # Scale coordinates back up if the display image was resized
            scale = img.shape[1] / disp_img.shape[1]
            x, y, w, h = int(x*scale), int(y*scale), int(w*scale), int(h*scale)
            
            # 1. Crop exactly what you selected
            crop = img[y:y+h, x:x+w]
            
            # 2. Force it to 64x64 so it won't break Phase 2 later
            crop_64x64 = cv2.resize(crop, (64, 64))
            
            # 3. Save the physical image file
            filename = f"{label}_{saved_count:03d}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, crop_64x64)
            
            print(f"Saved: {save_path}")
            saved_count += 1
            
        # Catch 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Cropping aborted by user.")
            break

    print(f"\nDone! Successfully saved {saved_count} physical images to {save_dir}")

if __name__ == "__main__":
    main()