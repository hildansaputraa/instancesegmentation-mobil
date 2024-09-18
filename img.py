import cv2
from yoloseg import YOLOSeg

# Initialize YOLOv5 Instance Segmentator
model_path = "models/model.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)

# Read image
img = "img2.jpg"

# Detect Objects
boxes, scores, class_ids, masks = yoloseg(img)

# Draw detections
combined_img = yoloseg.draw_masks(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("detected_objects.jpg", combined_img)
cv2.waitKey(0)
