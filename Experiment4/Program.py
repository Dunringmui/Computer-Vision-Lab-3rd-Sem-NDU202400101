import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "opencv_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def canny_edge_detection(image_path):
    """Perform Canny edge detection."""
    
    print("\n" + "="*60)
    print("  CANNY EDGE DETECTION")
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
    
    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
    print("✓ Gaussian blur applied")
    
    # Apply Canny with different thresholds
    edges_low = cv2.Canny(img_blur, 30, 100)
    edges_med = cv2.Canny(img_blur, 50, 150)
    edges_high = cv2.Canny(img_blur, 100, 200)
    
    print("✓ Canny edge detection applied (3 thresholds)")
    
    # Display results
    cv2.imshow('Original', img)
    cv2.imshow('Grayscale', img_gray)
    cv2.imshow('Edges - Low (30,100)', edges_low)
    cv2.imshow('Edges - Medium (50,150)', edges_med)
    cv2.imshow('Edges - High (100,200)', edges_high)
    
    # Create edge overlays
    overlay = img.copy()
    overlay[edges_med != 0] = [0, 255, 0]  # Green edges
    cv2.imshow('Edge Overlay', overlay)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Canny Edge Detection', fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_blur, cmap='gray')
    axes[0, 2].set_title('Blurred')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(edges_low, cmap='gray')
    axes[1, 0].set_title('Low Threshold (30,100)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edges_med, cmap='gray')
    axes[1, 1].set_title('Medium Threshold (50,150)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(edges_high, cmap='gray')
    axes[1, 2].set_title('High Threshold (100,200)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    cv2.imwrite(f"{OUTPUT_DIR}/edges_low.jpg", edges_low)
    cv2.imwrite(f"{OUTPUT_DIR}/edges_medium.jpg", edges_med)
    cv2.imwrite(f"{OUTPUT_DIR}/edges_high.jpg", edges_high)
    cv2.imwrite(f"{OUTPUT_DIR}/overlay.jpg", overlay)
    plt.savefig(f"{OUTPUT_DIR}/canny_comparison.png", dpi=150)
    
    print(f"\n✓ Results saved to '{OUTPUT_DIR}/'")
    print("\nEdge pixels detected:")
    print(f"  Low:    {np.count_nonzero(edges_low):,}")
    print(f"  Medium: {np.count_nonzero(edges_med):,}")
    print(f"  High:   {np.count_nonzero(edges_high):,}")
    
    print("\nClose windows to exit...")
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("\n" + "="*60)
    print("  CANNY EDGE DETECTION")
    print("="*60)
    
    image_path = input("\nEnter image path: ").strip()
    
    if image_path:
        canny_edge_detection(image_path)
        print("\nProgram completed!")
    else:
        print("Error: No image path provided!")


if __name__ == "__main__":
    main()