import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "opencv_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def histogram_analysis(image_path):
    """Compute histogram and perform histogram equalization."""
    
    print("\n" + "="*60)
    print("  HISTOGRAM ANALYSIS AND EQUALIZATION")
    print("="*60)
    
    # Load and convert to grayscale
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found!")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image!")
        return
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"✓ Image loaded: {img_gray.shape[1]}x{img_gray.shape[0]}")
    
    # Histogram equalization
    img_eq = cv2.equalizeHist(img_gray)
    print("✓ Histogram equalization applied")
    
    # Calculate histograms
    hist_orig = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])
    
    # Display images
    cv2.imshow('Original Grayscale', img_gray)
    cv2.imshow('Equalized', img_eq)
    comparison = np.hstack((img_gray, img_eq))
    cv2.imshow('Comparison: Original | Equalized', comparison)
    
    # Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Histogram Analysis and Equalization', fontsize=14, fontweight='bold')
    
    # Original image and histogram
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title('Original Grayscale')
    axes[0, 0].axis('off')
    
    axes[0, 1].plot(hist_orig, color='blue')
    axes[0, 1].set_title('Original Histogram')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xlim([0, 256])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Equalized image and histogram
    axes[1, 0].imshow(img_eq, cmap='gray')
    axes[1, 0].set_title('Histogram Equalized')
    axes[1, 0].axis('off')
    
    axes[1, 1].plot(hist_eq, color='red')
    axes[1, 1].set_title('Equalized Histogram')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    cv2.imwrite(f"{OUTPUT_DIR}/original_gray.jpg", img_gray)
    cv2.imwrite(f"{OUTPUT_DIR}/equalized.jpg", img_eq)
    cv2.imwrite(f"{OUTPUT_DIR}/comparison.jpg", comparison)
    plt.savefig(f"{OUTPUT_DIR}/histogram_plot.png", dpi=150)
    
    print(f"\n✓ Results saved to '{OUTPUT_DIR}/'")
    print("\nStatistics:")
    print(f"Original  - Mean: {np.mean(img_gray):.1f}, Std: {np.std(img_gray):.1f}")
    print(f"Equalized - Mean: {np.mean(img_eq):.1f}, Std: {np.std(img_eq):.1f}")
    
    print("\nClose windows to exit...")
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("\n" + "="*60)
    print("  HISTOGRAM COMPUTATION AND EQUALIZATION")
    print("="*60)
    
    image_path = input("\nEnter image path: ").strip()
    
    if image_path:
        histogram_analysis(image_path)
        print("\nProgram completed!")
    else:
        print("Error: No image path provided!")


if __name__ == "__main__":
    main()