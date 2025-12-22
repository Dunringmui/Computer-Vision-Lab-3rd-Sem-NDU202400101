import cv2
import torch
from ultralytics import YOLO
import numpy as np

# 1. SETUP
img_path = "sample.jpg"
output_path = "output.jpg"
image = cv2.imread(img_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# --- 2. ACCURATE HAAR CASCADE (Face Only) ---
# Increased minNeighbors to 12+ to kill the "ghost" box on the dog
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=12)

haar_result = image.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(haar_result, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(haar_result, "Haar Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
# Using YOLOv8m (Medium) at high resolution (imgsz=1280) for perfect boundaries
model = YOLO("yolov8m.pt") 
results = model.predict(source=image, imgsz=1280, conf=0.6, save=False)
yolo_result = results[0].plot()

# --- 4. COMBINE, RESIZE, AND SAVE ---
# Stack images side-by-side
combined = np.hstack((haar_result, yolo_result))

scale_percent = 70
width = int(combined.shape[1] * scale_percent / 100)
height = int(combined.shape[0] * scale_percent / 100)
dim = (width, height)

# INTER_AREA is best for shrinking images without losing sharpness
final_output = cv2.resize(combined, dim, interpolation=cv2.INTER_AREA)

# Save to disk
cv2.imwrite(output_path, final_output)
print(f"Success! Accurate comparison saved as: {output_path}")

# Display
cv2.imshow("Output", final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()