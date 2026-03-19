import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def test_preprocessing_filters(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- ACTION 1: ZERO-ORDER DERIVATIVE (Gaussian) ---
    blurred = cv2.GaussianBlur(image, (3,9), 0)
    blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    # --- ACTION 2: FIRST-ORDER DERIVATIVES (Sobel) ---
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

    # 1. Calculate them in PARALLEL on the blurred image
    sobel_horizontal = cv2.filter2D(blurred, cv2.CV_64F, kernel_h)
    sobel_vertical = cv2.filter2D(blurred, cv2.CV_64F, kernel_v)

    # 2. Combine them using the L2 Norm (Pythagorean theorem)
    sobel_combined = cv2.magnitude(sobel_horizontal, sobel_vertical)

    # Convert the mathematical outputs back to 8-bit visual images (0-255)
    visual_h = cv2.convertScaleAbs(sobel_horizontal)
    visual_v = cv2.convertScaleAbs(sobel_vertical)
    visual_c = cv2.convertScaleAbs(sobel_combined)

    # --- PLOT THE RESULTS ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('1. Original BGR')
    axes[0].axis('off')

    axes[1].imshow(blurred, cmap='gray')
    axes[1].set_title('2. Gaussian Blur')
    axes[1].axis('off')

    axes[2].imshow(visual_h, cmap='gray')
    axes[2].set_title('3. Horizontal Sobel')
    axes[2].axis('off')

    axes[3].imshow(visual_v, cmap='gray')
    axes[3].set_title('4. Vertical Sobel')
    axes[3].axis('off')

    axes[4].imshow(visual_c, cmap='gray') # FIXED: Plotted to axes[4]
    axes[4].set_title('5. L2 Norm (Unified Edges)')
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_image = os.path.join(os.getcwd(), "Test1.jpg")
    test_preprocessing_filters(test_image)