import time
import torch
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('models/modeltransfer.pt')  # Ubah dengan file modelmu

# Fungsi untuk menghitung jarak antara dua bounding box
def calculate_distance(box1, box2):
    # Mengambil koordinat tengah dari kedua box
    center_box1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center_box2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    
    # Menghitung jarak Euclidean antara dua titik
    distance = ((center_box1[0] - center_box2[0]) ** 2 + (center_box1[1] - center_box2[1]) ** 2) ** 0.5
    return distance

# Open webcam
cap = cv2.VideoCapture("video1.mp4")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop over frames from the webcam
prev_time = 0  # Inisialisasi waktu sebelumnya untuk menghitung FPS

while True:
    start_time = time.time()  # Waktu mulai frame ini

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # YOLO inference on the frame
    results = model(frame)
    
    # Get bounding boxes and class names
    boxes = results[0].boxes  # Bounding boxes
    names = results[0].names  # Class names
    
    # Find roadboundary and another object
    roadboundary_box = None
    object_box = None
    
    for i, box in enumerate(boxes):
        class_name = names[int(box.cls)]
        if class_name == "roadboundary":
            roadboundary_box = box.xyxy[0].cpu().numpy()  # Koordinat bounding box roadboundary
        elif class_name == "car":  # Atau object lain yang ingin kamu deteksi
            object_box = box.xyxy[0].cpu().numpy()  # Koordinat bounding box object
    
    # Jika bounding box "roadboundary" dan "object" ditemukan
    if roadboundary_box is not None and object_box is not None:
        # Hitung jarak antara bounding box
        distance = calculate_distance(roadboundary_box, object_box)
        
        # Jika jaraknya kurang dari 3 cm (disesuaikan dalam unit pixel)
        if distance < 30:  # 30 pixel sebagai contoh
            cv2.putText(frame, "WARNING: Too Close to Roadboundary!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Visualize the results
    annotated_frame = results[0].plot()

    # Hitung FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Tampilkan FPS di atas frame
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with YOLO predictions, warnings, and FPS
    cv2.imshow('Webcam YOLO', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
