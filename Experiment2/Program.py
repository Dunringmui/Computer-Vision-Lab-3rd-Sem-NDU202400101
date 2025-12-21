import cv2
import numpy as np
import os

OUTPUT_DIR = "opencv_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def scaling_transformation(img):
    """Apply scaling transformations."""
    print(f"\n{'='*50}\nSCALING TRANSFORMATION\n{'='*50}")
    
    h, w = img.shape[:2]
    print(f"Original: {w}x{h}")
    
    scaled_up = cv2.resize(img, None, fx=1.5, fy=1.5)
    scaled_down = cv2.resize(img, None, fx=0.5, fy=0.5)
    scaled_fixed = cv2.resize(img, (400, 300))
    
    print("✓ Scaled Up (1.5x), Down (0.5x), Fixed (400x300)")
    
    cv2.imshow('Original', img)
    cv2.imshow('Scaled Up 1.5x', scaled_up)
    cv2.imshow('Scaled Down 0.5x', scaled_down)
    cv2.imshow('Fixed Size 400x300', scaled_fixed)
    
    cv2.imwrite(f"{OUTPUT_DIR}/scaled_up.jpg", scaled_up)
    cv2.imwrite(f"{OUTPUT_DIR}/scaled_down.jpg", scaled_down)
    
    print("Press any key...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def translation_transformation(img):
    """Apply translation transformations."""
    print(f"\n{'='*50}\nTRANSLATION TRANSFORMATION\n{'='*50}")
    
    h, w = img.shape[:2]
    
    # Shift right(100) and down(50)
    M1 = np.float32([[1, 0, 100], [0, 1, 50]])
    trans1 = cv2.warpAffine(img, M1, (w, h))
    
    # Shift left(-80) and up(-60)
    M2 = np.float32([[1, 0, -80], [0, 1, -60]])
    trans2 = cv2.warpAffine(img, M2, (w, h))
    
    # Horizontal shift only
    M3 = np.float32([[1, 0, 150], [0, 1, 0]])
    trans3 = cv2.warpAffine(img, M3, (w, h))
    
    print("✓ Right+Down (100,50), Left+Up (-80,-60), Horizontal (150)")
    
    cv2.imshow('Original', img)
    cv2.imshow('Right+Down', trans1)
    cv2.imshow('Left+Up', trans2)
    cv2.imshow('Horizontal', trans3)
    
    cv2.imwrite(f"{OUTPUT_DIR}/translated_rightdown.jpg", trans1)
    cv2.imwrite(f"{OUTPUT_DIR}/translated_leftup.jpg", trans2)
    
    print("Press any key...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotation_transformation(img):
    """Apply rotation transformations."""
    print(f"\n{'='*50}\nROTATION TRANSFORMATION\n{'='*50}")
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotate 45, 90, 180 degrees
    M1 = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_45 = cv2.warpAffine(img, M1, (w, h))
    
    M2 = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated_90 = cv2.warpAffine(img, M2, (w, h))
    
    M3 = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated_180 = cv2.warpAffine(img, M3, (w, h))
    
    # Rotate with scale
    M4 = cv2.getRotationMatrix2D(center, 30, 1.3)
    rotated_scaled = cv2.warpAffine(img, M4, (w, h))
    
    print("✓ Rotated 45°, 90°, 180°, and 30° with 1.3x scale")
    
    cv2.imshow('Original', img)
    cv2.imshow('Rotated 45°', rotated_45)
    cv2.imshow('Rotated 90°', rotated_90)
    cv2.imshow('Rotated 180°', rotated_180)
    cv2.imshow('Rotated 30° + Scale', rotated_scaled)
    
    cv2.imwrite(f"{OUTPUT_DIR}/rotated_45.jpg", rotated_45)
    cv2.imwrite(f"{OUTPUT_DIR}/rotated_90.jpg", rotated_90)
    cv2.imwrite(f"{OUTPUT_DIR}/rotated_scaled.jpg", rotated_scaled)
    
    print("Press any key...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def all_transformations(img):
    """Display all transformations together."""
    print(f"\n{'='*50}\nALL TRANSFORMATIONS\n{'='*50}")
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Apply one of each transformation
    scaled = cv2.resize(img, None, fx=0.7, fy=0.7)
    scaled = cv2.copyMakeBorder(scaled, 
                                (h - scaled.shape[0]) // 2,
                                (h - scaled.shape[0]) // 2,
                                (w - scaled.shape[1]) // 2,
                                (w - scaled.shape[1]) // 2,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    M_trans = np.float32([[1, 0, 80], [0, 1, 60]])
    translated = cv2.warpAffine(img, M_trans, (w, h))
    
    M_rot = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(img, M_rot, (w, h))
    
    print("✓ Displaying: Original, Scaled, Translated, Rotated")
    
    cv2.imshow('1. Original', img)
    cv2.imshow('2. Scaled 0.7x', scaled)
    cv2.imshow('3. Translated (80,60)', translated)
    cv2.imshow('4. Rotated 45°', rotated)
    
    print("Press any key...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("\n" + "="*50)
    print("  GEOMETRIC TRANSFORMATIONS - OpenCV")
    print("="*50)
    
    img_path = input("\nEnter image path: ").strip()
    
    if not os.path.exists(img_path):
        print(f"Error: '{img_path}' not found!")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not load image!")
        return
    
    print(f"✓ Image loaded: {img.shape[1]}x{img.shape[0]}")
    
    while True:
        print(f"\n{'='*50}")
        print("1. Scaling")
        print("2. Translation")
        print("3. Rotation")
        print("4. All Transformations")
        print("5. Exit")
        print("="*50)
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == '1':
            scaling_transformation(img)
        elif choice == '2':
            translation_transformation(img)
        elif choice == '3':
            rotation_transformation(img)
        elif choice == '4':
            all_transformations(img)
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()