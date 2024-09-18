import cv2
import numpy as np
from ultralytics import YOLO

def predict_on_image(model, img, conf):
    result = model(img, conf=conf)[0]

    cls = result.boxes.cls.cpu().numpy()  # (N, 1)
    probs = result.boxes.conf.cpu().numpy()  # (N, 1)
    boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # (N, H, W)
        masks = np.moveaxis(masks, 0, -1)  # (H, W, N)
        masks = np.moveaxis(masks, -1, 0)  # (N, H, W)
    else:
        masks = np.array([])  # No masks available

    return boxes, masks, cls, probs

def overlay(image, mask, color, alpha, resize=None):
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    return image_combined

# Load YOLO model
model = YOLO('models/best.pt')

# Initialize video capture and writer
input_video = 'video6.mp4'
output_video = 'output_video.mp4'

cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict segmentation
    boxes, masks, cls, probs = predict_on_image(model, frame, conf=0.55)

    # Overlay masks on the frame
    frame_with_masks = np.copy(frame)
    if masks.size > 0:  # Check if masks are available
        for mask_i in masks:
            frame_with_masks = overlay(frame_with_masks, mask_i, color=(0, 255, 0), alpha=0.3)

    # Write the frame with masks to the output video
    #out.write(frame_with_masks)

    # Display the frame with masks
    cv2.imshow('Video with Masks', frame_with_masks)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
