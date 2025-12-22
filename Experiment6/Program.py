import cv2
import numpy as np


# Load Image (COLOR)
image_path = "apple.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Resize for uniform comparison
image = cv2.resize(image, (400, 300))

# Smoothing Filter (Averaging)
kernel = np.ones((5, 5), np.float32) / 25
smoothed = cv2.filter2D(image, -1, kernel)

# Motion Blur Simulation (COLOR)
size = 15
motion_kernel = np.zeros((size, size))
motion_kernel[int((size - 1) / 2), :] = np.ones(size)
motion_kernel = motion_kernel / size

motion_blur = cv2.filter2D(image, -1, motion_kernel)

# Combine Images Side by Side
comparison_images = np.hstack((image, smoothed, motion_blur))

# Create Label Area (Outside Image)
label_height = 50
label_area = np.ones(
    (label_height, comparison_images.shape[1], 3),
    dtype=np.uint8
) * 255  # white background

# Image width
w = image.shape[1]

# Add text labels centered below each image
cv2.putText(label_area, "Original Image", (w//2 - 80, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

cv2.putText(label_area, "Filtered Image", (w + w//2 - 80, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

cv2.putText(label_area, "Motion Blur Image", (2*w + w//2 - 110, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Stack Images + Labels Vertically
final_output = np.vstack((comparison_images, label_area))

# Display Output
cv2.imshow("Output", final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save Output Image
output_path = "Output.jpg"
cv2.imwrite(output_path, final_output)

print(f"Comparison image saved as {output_path}")
