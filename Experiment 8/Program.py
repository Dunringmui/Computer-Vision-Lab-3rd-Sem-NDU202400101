import numpy as np
import cv2

def create_gabor_filter_bank():
    filters = []
    ksize = 31 
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

def process_image(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8U, kern)
        np.maximum(accum, fimg, accum)
    return accum

def add_bottom_label(image, text):
    """Adds text labels at the very bottom of the image segment."""
    output = image.copy()
    height, width = image.shape[:2]
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.8
    thickness = 2
    
    # Calculate text size to center it horizontally
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    tx = (width - text_size[0]) // 2
    
    # Set vertical position to the bottom (height minus 20 pixels)
    ty = height - 20 
    
    # Draw black outline for readability on white backgrounds
    cv2.putText(output, text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Draw white text on top
    cv2.putText(output, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return output

# 1. Load and Resize
img = cv2.imread('apple.jpg')
scale = 0.5
img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

# 2. Gabor Filtering
filters = create_gabor_filter_bank()
res1 = process_image(gray, filters)

# 3. Automatic Thresholding (Otsu's Method)
_, binary_output = cv2.threshold(res1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. Morphological Cleaning
kernel = np.ones((5,5), np.uint8)
final_seg = cv2.morphologyEx(binary_output, cv2.MORPH_CLOSE, kernel)

# --- PREPARE COMPARISON ---
res1_bgr = cv2.cvtColor(res1, cv2.COLOR_GRAY2BGR)
seg_bgr = cv2.cvtColor(final_seg, cv2.COLOR_GRAY2BGR)

# Apply Labels at the Bottom
orig_lab = add_bottom_label(img_small, "Original")
filt_lab = add_bottom_label(res1_bgr, "Gabor Response")
seg_lab = add_bottom_label(seg_bgr, "Segmentation")

# Concatenate horizontally
comparison = np.concatenate((orig_lab, filt_lab, seg_lab), axis=1)

# 5. Display and Save
cv2.imshow('Output', comparison)
cv2.imwrite('Output.jpg', comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()