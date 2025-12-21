import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "opencv_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def spatial_filtering(image_path):
    """Apply smoothing and sharpening filters."""
    
    print("\n" + "="*60)
    print("  SPATIAL DOMAIN FILTERING")
    print("="*60)
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found!")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image!")
        return
    
    print(f"✓ Image loaded: {img.shape[1]}x{img.shape[0]}")
    
    # ===== SMOOTHING FILTERS =====
    print("\n1. Applying Smoothing Filters...")
    
    # Average (Box) filter
    avg_filter = cv2.blur(img, (5, 5))
    print("   ✓ Average filter (5x5)")
    
    # Gaussian filter
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    print("   ✓ Gaussian filter (5x5)")
    
    # Median filter
    median = cv2.medianBlur(img, 5)
    print("   ✓ Median filter (5x5)")
    
    # Bilateral filter (edge-preserving)
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    print("   ✓ Bilateral filter (edge-preserving)")
    
    # ===== SHARPENING FILTERS =====
    print("\n2. Applying Sharpening Filters...")
    
    # Sharpening kernel
    kernel_sharp = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
    sharpened1 = cv2.filter2D(img, -1, kernel_sharp)
    print("   ✓ Basic sharpening kernel")
    
    # Unsharp masking
    gaussian_blur = cv2.GaussianBlur(img, (9, 9), 10.0)
    unsharp = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)
    print("   ✓ Unsharp masking")
    
    # Laplacian sharpening
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    laplacian_sharp = cv2.add(gray, laplacian)
    print("   ✓ Laplacian sharpening")
    
    # ===== DISPLAY RESULTS =====
    print("\n3. Displaying results...")
    
    cv2.imshow('Original', img)
    cv2.imshow('Smoothing - Average', avg_filter)
    cv2.imshow('Smoothing - Gaussian', gaussian)
    cv2.imshow('Smoothing - Median', median)
    cv2.imshow('Smoothing - Bilateral', bilateral)
    cv2.imshow('Sharpening - Basic', sharpened1)
    cv2.imshow('Sharpening - Unsharp Mask', unsharp)
    cv2.imshow('Sharpening - Laplacian', laplacian_sharp)
    
    # ===== CREATE COMPARISON PLOT =====
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Spatial Domain Filtering - Smoothing and Sharpening', 
                 fontsize=14, fontweight='bold')
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    avg_rgb = cv2.cvtColor(avg_filter, cv2.COLOR_BGR2RGB)
    gaussian_rgb = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
    median_rgb = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
    bilateral_rgb = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)
    sharp1_rgb = cv2.cvtColor(sharpened1, cv2.COLOR_BGR2RGB)
    unsharp_rgb = cv2.cvtColor(unsharp, cv2.COLOR_BGR2RGB)
    
    # Plot images
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(avg_rgb)
    axes[0, 1].set_title('Average Filter (5x5)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gaussian_rgb)
    axes[0, 2].set_title('Gaussian Filter (5x5)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(median_rgb)
    axes[1, 0].set_title('Median Filter (5x5)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(bilateral_rgb)
    axes[1, 1].set_title('Bilateral Filter')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(sharp1_rgb)
    axes[1, 2].set_title('Basic Sharpening')
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(unsharp_rgb)
    axes[2, 0].set_title('Unsharp Masking')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(laplacian_sharp, cmap='gray')
    axes[2, 1].set_title('Laplacian Sharpening')
    axes[2, 1].axis('off')
    
    # Info panel
    axes[2, 2].axis('off')
    info_text = """
    SMOOTHING FILTERS:
    • Average: Simple blur
    • Gaussian: Weighted blur
    • Median: Noise reduction
    • Bilateral: Edge-preserving
    
    SHARPENING FILTERS:
    • Basic: Edge enhancement
    • Unsharp Mask: Detail boost
    • Laplacian: Edge detection
    """
    axes[2, 2].text(0.1, 0.5, info_text, fontsize=9, 
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # ===== SAVE RESULTS =====
    print("\n4. Saving results...")
    
    cv2.imwrite(f"{OUTPUT_DIR}/original.jpg", img)
    cv2.imwrite(f"{OUTPUT_DIR}/smooth_average.jpg", avg_filter)
    cv2.imwrite(f"{OUTPUT_DIR}/smooth_gaussian.jpg", gaussian)
    cv2.imwrite(f"{OUTPUT_DIR}/smooth_median.jpg", median)
    cv2.imwrite(f"{OUTPUT_DIR}/smooth_bilateral.jpg", bilateral)
    cv2.imwrite(f"{OUTPUT_DIR}/sharp_basic.jpg", sharpened1)
    cv2.imwrite(f"{OUTPUT_DIR}/sharp_unsharp.jpg", unsharp)
    cv2.imwrite(f"{OUTPUT_DIR}/sharp_laplacian.jpg", laplacian_sharp)
    plt.savefig(f"{OUTPUT_DIR}/filtering_comparison.png", dpi=150, bbox_inches='tight')
    
    print(f"✓ All results saved to '{OUTPUT_DIR}/'")
    print("\n" + "="*60)
    print("Filtering complete!")
    print("="*60)
    print("\nClose windows to exit...")
    
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("\n" + "="*60)
    print("  SPATIAL DOMAIN FILTERING")
    print("  Smoothing and Sharpening")
    print("="*60)
    
    image_path = input("\nEnter image path: ").strip()
    
    if image_path:
        spatial_filtering(image_path)
        print("\nProgram completed!")
    else:
        print("Error: No image path provided!")


if __name__ == "__main__":
    main()