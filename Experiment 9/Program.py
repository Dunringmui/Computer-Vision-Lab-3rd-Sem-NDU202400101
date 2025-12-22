import cv2
import numpy as np
import matplotlib.pyplot as plt

def perform_segmentation(path):
    img = cv2.imread(path)
    if img is None: return print("Error: Image not found!")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Classical Methods
    # Otsu's Thresholding (Automatic binary)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Watershed (Simple version)
    ret, markers = cv2.connectedComponents(otsu)
    watershed_img = img.copy()
    markers = cv2.watershed(watershed_img, markers)
    watershed_img[markers == -1] = [255, 0, 0] # Red boundaries

    # GrabCut (Interactive/Rectangular)
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (10, 10, img.shape[1]-10, img.shape[0]-10) # Assumes central object
    cv2.grabCut(img, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
    grabcut = img * np.where((mask==2)|(mask==0), 0, 1).astype('uint8')[:,:,np.newaxis]

    # 2. Learning-Based Methods
    # K-Means Clustering
    pixels = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    kmeans = np.uint8(centers)[labels.flatten()].reshape(img.shape)

    # Mean Shift
    mean_shift = cv2.pyrMeanShiftFiltering(img, 21, 51)

    # 3. Visualization
    results = {
        "Original": img_rgb,
        "Thresholding": otsu, 
        "Watershed": cv2.cvtColor(watershed_img, cv2.COLOR_BGR2RGB),
        "K-Means": cv2.cvtColor(kmeans, cv2.COLOR_BGR2RGB),
    }

    plt.figure(figsize=(12, 8))
    for i, (title, res) in enumerate(results.items()):
        plt.subplot(2, 3, i+1)
        plt.imshow(res, cmap='gray' if title == "Thresholding" else None)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()

    # -------------------------------
    # SAVE OUTPUT (ADDED)
    # -------------------------------
    output_path = "Output.jpg"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Output saved as {output_path}")

    plt.show()

if __name__ == "__main__":
    path = input("Enter image path: ").strip()
    perform_segmentation(path)
