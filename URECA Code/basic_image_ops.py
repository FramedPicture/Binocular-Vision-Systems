"""
Basic Image Operations - Module 1 Exercise 1
Learn the fundamentals of loading, displaying, and manipulating images with OpenCV
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_and_display_image(image_path):
    """
    Load an image and display it using both OpenCV and Matplotlib
    """
    if not isinstance(image_path, Path):
        image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = DATA_DIR / image_path

    # Load image using OpenCV (BGR format)
    img_bgr = cv2.imread(str(image_path))
    
    if img_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram statistics
    mean_intensity = np.mean(img_gray)
    std_intensity = np.std(img_gray)
    
    # Display images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image (RGB)\nColor channels: Red, Green, Blue')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title(f'Grayscale Image\n0=Black, 255=White\nMean: {mean_intensity:.1f}')
    plt.axis('off')
    
    # Show histogram
    plt.subplot(1, 3, 3)
    plt.hist(img_gray.flatten(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.axvline(mean_intensity, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_intensity:.1f}')
    plt.title('Pixel Intensity Histogram\nDistribution of brightness values')
    plt.xlabel('Pixel Intensity (0=Dark → 255=Bright)')
    plt.ylabel('Frequency (Pixel Count)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return img_rgb, img_gray

def apply_basic_filters(img_gray):
    """
    Apply basic image filters to understand image processing
    """
    # Gaussian blur
    gaussian_blur = cv2.GaussianBlur(img_gray, (15, 15), 0)
    
    # Median blur
    median_blur = cv2.medianBlur(img_gray, 15)
    
    # Bilateral filter (edge-preserving)
    bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
    
    # Sobel edge detection (gradient in X direction)
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.absolute(sobel_x)
    
    # Sobel edge detection (gradient in Y direction)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.absolute(sobel_y)
    
    # Combined Sobel
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Display results
    plt.figure(figsize=(20, 10))
    
    images = [
        (img_gray, 'Original\nUnprocessed grayscale'),
        (gaussian_blur, 'Gaussian Blur\nReduces noise & detail'),
        (median_blur, 'Median Blur\nRemoves salt-pepper noise'),
        (bilateral, 'Bilateral Filter\nSmooths while preserving edges'),
        (sobel_x, 'Sobel X (Horizontal)\nDetects vertical edges'),
        (sobel_y, 'Sobel Y (Vertical)\nDetects horizontal edges'),
        (sobel_combined, 'Sobel Combined\nAll edge directions')
    ]
    
    for i, (img, title) in enumerate(images):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return gaussian_blur, bilateral, sobel_combined

def create_sample_image():
    """
    Create a sample image with geometric shapes for testing
    """
    # Create a blank image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Draw some shapes
    cv2.rectangle(img, (50, 50), (200, 150), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(img, (350, 100), 75, (0, 255, 0), -1)  # Green circle
    cv2.fillPoly(img, [np.array([[450, 50], [550, 50], [500, 150]])], (0, 0, 255))  # Red triangle
    
    # Add some noise
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img_noisy = cv2.add(img, noise)
    
    return img, img_noisy

def main():
    """
    Main function to demonstrate basic image operations
    """
    print("=== Basic Image Operations Demo ===")
    print("1. Creating sample images...")
    
    # Create sample images
    clean_img, noisy_img = create_sample_image()
    
    # Save sample images
    cv2.imwrite(str(DATA_DIR / 'sample_clean.png'), clean_img)
    cv2.imwrite(str(DATA_DIR / 'sample_noisy.png'), noisy_img)
    
    print("2. Loading and displaying images...")
    
    # Try to load a sample image (you can replace this with your own image path)
    try:
        result = load_and_display_image('sample_clean.png')
        if result is not None:
            img_rgb, img_gray = result
            
            print("3. Applying basic filters...")
            apply_basic_filters(img_gray)
            print("4. Trying with noisy image...")
            noisy_result = load_and_display_image('sample_noisy.png')
            if noisy_result is not None:
                noisy_rgb, noisy_gray = noisy_result
                apply_basic_filters(noisy_gray)
                apply_basic_filters(noisy_gray)
        
    except Exception as e:
        print(f"Error processing images: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install opencv-python matplotlib numpy")
    
    print("\n=== Exercise Complete ===")
    print("Key Takeaways:")
    print("- Images are represented as numpy arrays")
    print("- OpenCV loads images in BGR format, matplotlib expects RGB")
    print("- Gaussian blur reduces noise but also reduces detail")
    print("- Bilateral filter preserves edges while reducing noise")
    print("- Sobel filters detect edges by computing gradients")

if __name__ == "__main__":
    main()