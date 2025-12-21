import cv2
import numpy as np

def process_motion_blur(image_path, size=40):
    # 1. Load the original image
    original = cv2.imread(image_path)
    
    if original is None:
        print(f"Error: Could not find '{image_path}'")
        return

    # 2. Create the Motion Blur Kernel (Horizontal)
    # A horizontal line of 1s in the center of the matrix
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel /= size  # Normalize to maintain original image brightness

    # 3. Apply Convolution using the kernel
    blurred = cv2.filter2D(original, -1, kernel)

    # 4. Create the side-by-side comparison
    # np.hstack joins the original and blurred arrays horizontally
    comparison = np.hstack((original, blurred))

    # 5. Store the output (the side-by-side image)
    output_path = "Output.jpg"
    cv2.imwrite(output_path, comparison)
    print(f"Comparison image saved successfully as: {output_path}")

    # 6. Display the results in a window
    cv2.imshow('Original (Left) vs Motion Blur (Right)', comparison)
    
    print("Press any key on the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Execute the function
process_motion_blur("sample_image.jpg", size=40)