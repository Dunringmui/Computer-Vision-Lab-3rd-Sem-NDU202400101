import cv2
import numpy as np

# 1. Load the image
img = cv2.imread('apple.jpg')
if img is None:
    print("Error: Could not load the image.")
    exit()

# 2. Resize 
scale = 1
img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

# --- Helper function to add a label with a background for visibility ---
def add_labeled_overlay(image, text):
    output = image.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8
    color = (0, 0, 0) # Black text
    thickness = 1
    pos = (15, 35) # Top-left position
    
    # Add a white rectangle background so the text is always visible
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(output, (pos[0]-5, pos[1]-h-5), (pos[0]+w+5, pos[1]+5), (255, 255, 255), -1)
    
    cv2.putText(output, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
    return output

# --- PROCESS IMAGES ---

# Original
original_labeled = add_labeled_overlay(img_small, "Original")

# Harris Corner Detection
gray_float = np.float32(gray)
harris_dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
harris_dst = cv2.dilate(harris_dst, None)
harris_out = img_small.copy()
harris_out[harris_dst > 0.01 * harris_dst.max()] = [0, 0, 255]
harris_labeled = add_labeled_overlay(harris_out, "Harris")


# --- COMBINE AND STORE ---
# Horizontal concatenation: [Original | Harris]
combined_output = np.concatenate((original_labeled, harris_labeled), axis=1)

cv2.imwrite('Output.jpg', combined_output)
cv2.imshow('Final Detection Results', combined_output)

print("Image saved as 'Output.jpg'")
cv2.waitKey(0)

cv2.destroyAllWindows()

