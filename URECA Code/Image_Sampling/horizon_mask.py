import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt

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

image_path = "Test1.jpg"


image_path = os.path.join(os.getcwd(), "Test1.jpg")
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found or cannot be loaded.")
    exit()
# img_edge = cv2.Canny(img,100,200)

cv2.imshow("img",img)
# cv2.imshow("img_edge",img_edge)

# img_edge2 = step_by_step_canny(img)
cv2.waitKey(0)

