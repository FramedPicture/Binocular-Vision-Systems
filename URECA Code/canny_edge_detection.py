"""
Canny Edge Detection - Module 1 Exercise 2
Implement and experiment with the Canny edge detection algorithm
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def canny_edge_detection(image, low_threshold=50, high_threshold=150, blur_kernel=5):
    """
    Perform Canny edge detection on an image
    
    Args:
        image: Input grayscale image
        low_threshold: Lower threshold for edge detection
        high_threshold: Higher threshold for edge detection
        blur_kernel: Size of Gaussian blur kernel (must be odd)
    
    Returns:
        edges: Binary edge image
        blurred: Gaussian blurred image (intermediate step)
    """
    # Step 1: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    
    # Step 2: Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges, blurred

def compare_thresholds(image):
    """
    Compare different threshold combinations for Canny edge detection
    """
    # Different threshold combinations
    threshold_combinations = [
        (50, 100, "Low Sensitivity"),
        (50, 150, "Medium Sensitivity"),
        (100, 200, "High Sensitivity"),
        (30, 80, "Very Sensitive"),
    ]
    
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image\nInput for edge detection')
    plt.axis('off')
    
    # Apply Canny with different thresholds
    for i, (low, high, label) in enumerate(threshold_combinations):
        edges, _ = canny_edge_detection(image, low, high)
        edge_count = np.sum(edges > 0)
        
        plt.subplot(2, 3, i + 2)
        plt.imshow(edges, cmap='gray')
        plt.title(f'{label}\nThresholds: {low}/{high}\nEdges: {edge_count:,} pixels', fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def step_by_step_canny(image, low_threshold=50, high_threshold=150):
    """
    Show each step of the Canny edge detection process
    """
    # Step 1: Original image
    original = image.copy()
    
    # Step 2: Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Step 3: Gradient calculation (Sobel)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = np.uint8(gradient_magnitude * 255 / gradient_magnitude.max())
    
    # Step 4: Final Canny edges
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Display all steps
    plt.figure(figsize=(20, 10))
    
    images_and_titles = [
        (original, 'Step 1: Original Image\nInput grayscale image'),
        (blurred, 'Step 2: Gaussian Blur\nReduces noise for cleaner edges'),
        (gradient_magnitude, 'Step 3: Gradient Magnitude\nShows edge strength (brightness)'),
        (edges, 'Step 4: Final Edges\nAfter non-max suppression & hysteresis')
    ]
    
    for i, (img, title) in enumerate(images_and_titles):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title, fontsize=11)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return edges

def interactive_canny_demo(image):
    """
    Demonstrate how parameters affect edge detection
    """
    print("=== Interactive Canny Parameter Demo ===")
    print("Trying different parameter combinations...")
    
    # Test different blur kernel sizes
    blur_kernels = [3, 5, 7, 9]
    plt.figure(figsize=(20, 5))
    
    for i, kernel in enumerate(blur_kernels):
        edges, blurred = canny_edge_detection(image, blur_kernel=kernel)
        edge_count = np.sum(edges > 0)
        
        plt.subplot(1, 4, i + 1)
        plt.imshow(edges, cmap='gray')
        smoothness = ['Sharp/Noisy', 'Balanced', 'Smooth', 'Very Smooth'][i]
        plt.title(f'Kernel: {kernel}×{kernel}\n{smoothness}\nEdges: {edge_count:,}', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Effect of Gaussian Blur Kernel Size\n(Larger kernel = smoother but less detail)', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Test threshold ratios
    print("\nTesting different threshold ratios...")
    base_low = 50
    ratios = [2, 2.5, 3, 4]  # high = low * ratio
    
    plt.figure(figsize=(20, 5))
    
    for i, ratio in enumerate(ratios):
        high_thresh = int(base_low * ratio)
        edges, _ = canny_edge_detection(image, base_low, high_thresh)
        edge_count = np.sum(edges > 0)
        
        plt.subplot(1, 4, i + 1)
        plt.imshow(edges, cmap='gray')
        selectivity = ['Weak edges included', 'Moderate filtering', 'Strong filtering', 'Very selective'][i]
        plt.title(f'Ratio 1:{ratio}\nLow:{base_low} High:{high_thresh}\n{selectivity}\nEdges: {edge_count:,}', fontsize=9)
        plt.axis('off')
    
    plt.suptitle('Effect of Threshold Ratios\n(Recommended ratio 1:2 to 1:3 for most images)', fontsize=12)
    plt.tight_layout()
    plt.show()

def edge_statistics(image, edges):
    """
    Calculate and display statistics about detected edges
    """
    total_pixels = image.shape[0] * image.shape[1]
    edge_pixels = np.sum(edges > 0)
    edge_percentage = (edge_pixels / total_pixels) * 100
    
    print("=== Edge Detection Statistics ===")
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Edge pixels: {edge_pixels:,}")
    print(f"Edge percentage: {edge_percentage:.2f}%")
    
    return edge_percentage

def create_test_patterns():
    """
    Create test images with different characteristics
    """
    # Pattern 1: Simple geometric shapes
    img1 = np.zeros((300, 400), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (150, 150), 255, -1)
    cv2.circle(img1, (250, 100), 50, 128, -1)
    cv2.rectangle(img1, (300, 200), (380, 280), 200, 3)
    
    # Pattern 2: Noisy image
    img2 = img1.copy()
    noise = np.random.normal(0, 25, img2.shape).astype(np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Pattern 3: Textured image
    img3 = np.zeros((300, 400), dtype=np.uint8)
    for i in range(0, 400, 20):
        cv2.line(img3, (i, 0), (i, 300), 255, 2)
    for j in range(0, 300, 20):
        cv2.line(img3, (0, j), (400, j), 128, 1)
    
    return img1, img2, img3

def main():
    """
    Main function to demonstrate Canny edge detection
    """
    print("=== Canny Edge Detection Demo ===")
    
    # Create test patterns
    print("1. Creating test patterns...")
    pattern1, pattern2, pattern3 = create_test_patterns()
    
    # Save test patterns
    cv2.imwrite(str(DATA_DIR / 'pattern_simple.png'), pattern1)
    cv2.imwrite(str(DATA_DIR / 'pattern_noisy.png'), pattern2)
    cv2.imwrite(str(DATA_DIR / 'pattern_textured.png'), pattern3)
    
    patterns = [
        (pattern1, "Simple Geometric Shapes"),
        (pattern2, "Noisy Image"),
        (pattern3, "Textured Image")
    ]
    
    for image, description in patterns:
        print(f"\n2. Processing: {description}")
        
        # Basic Canny edge detection
        edges, blurred = canny_edge_detection(image)
        
        # Show step-by-step process
        print("   - Showing step-by-step Canny process...")
        step_by_step_canny(image)
        
        # Compare different thresholds
        print("   - Comparing different thresholds...")
        compare_thresholds(image)
        
        # Interactive parameter demo
        print("   - Interactive parameter demonstration...")
        interactive_canny_demo(image)
        
        # Calculate statistics
        edge_percentage = edge_statistics(image, edges)
        
        print(f"   - Edge detection complete! {edge_percentage:.1f}% of pixels are edges.")
    
    print("\n=== Exercise Complete ===")
    print("Key Learnings:")
    print("- Lower thresholds detect more edges (including noise)")
    print("- Higher thresholds detect only strong edges")
    print("- The ratio between high and low thresholds is important")
    print("- Gaussian blur kernel size affects edge continuity")
    print("- Different image types require different parameter tuning")

if __name__ == "__main__":
    main()