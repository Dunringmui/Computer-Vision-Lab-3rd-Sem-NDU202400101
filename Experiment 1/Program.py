import cv2
import numpy as np
import os
from datetime import datetime

# Configuration
IMAGE_PATH = "sample_image.jpg"
OUTPUT_DIR = "opencv_output"

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def read_image_modes(image_path):
    """Read and display an image in multiple modes."""
    print(f"\n{'='*60}")
    print("IMAGE ACQUISITION - Multiple Reading Modes")
    print(f"{'='*60}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Read in different modes
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_unchanged = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    
    print(f"✓ Color (BGR): {img_color.shape}")
    print(f"✓ Grayscale: {img_gray.shape}")
    print(f"✓ Unchanged: {img_unchanged.shape}")
    print(f"✓ HSV: {img_hsv.shape}")
    
    # Display images
    cv2.imshow('Color (BGR)', img_color)
    cv2.imshow('Grayscale', img_gray)
    cv2.imshow('HSV', img_hsv)
    
    # Save images
    cv2.imwrite(f"{OUTPUT_DIR}/image_color.jpg", img_color)
    cv2.imwrite(f"{OUTPUT_DIR}/image_grayscale.jpg", img_gray)
    cv2.imwrite(f"{OUTPUT_DIR}/image_hsv.jpg", img_hsv)
    print(f"\n✓ Images saved to '{OUTPUT_DIR}/' directory")
    
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def capture_live_video(camera_index=0):
    """Capture live video from camera and save it."""
    print(f"\n{'='*60}")
    print("VIDEO ACQUISITION - Live Camera Capture")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    
    print(f"✓ Camera opened: {width}x{height} @ {fps} FPS")
    
    # Setup video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}/video_capture_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    print(f"✓ Recording to: {output_file}")
    print("\n*** IMPORTANT: Click on the video window first! ***")
    print("Controls: Q/ESC = quit | S = snapshot | G = grayscale")
    print("OR use Ctrl+C in terminal to stop")
    print(f"{'='*60}\n")
    
    frame_count = 0
    grayscale_mode = False
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Grayscale mode
        if grayscale_mode:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            cv2.putText(display_frame, "GRAYSCALE", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Frame info overlay
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        info = f"Frame: {frame_count} | Time: {elapsed:.1f}s"
        cv2.putText(display_frame, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Live Video', display_frame)
        out.write(frame)
        
        # Keyboard controls - make sure window is in focus!
        key = cv2.waitKey(30) & 0xFF  # Increased delay for better key detection
        if key == ord('q') or key == ord('Q') or key == 27:  # q, Q, or ESC
            print("\nStopping...")
            break
        elif key == ord('s') or key == ord('S'):
            cv2.imwrite(f"{OUTPUT_DIR}/snapshot_{timestamp}_{frame_count}.jpg", frame)
            print(f"✓ Snapshot saved: frame {frame_count}")
        elif key == ord('g') or key == ord('G'):
            grayscale_mode = not grayscale_mode
            print(f"✓ Grayscale: {'ON' if grayscale_mode else 'OFF'}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"Recording complete: {frame_count} frames in {elapsed:.2f}s")
    print(f"Saved: {output_file}")
    print(f"{'='*60}\n")

def main():
    """Main menu."""
    print("\n" + "="*60)
    print("OpenCV Image and Video Acquisition Demo")
    print("="*60)
    
    while True:
        print("\nMenu:")
        print("1. Read image in multiple modes")
        print("2. Capture live video from camera")
        print("3. Exit")

        choice = input("\nChoice (1-3): ").strip()

        if choice == '1':
            read_image_modes(IMAGE_PATH)
        elif choice == '2':
            capture_live_video()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":

    main()
