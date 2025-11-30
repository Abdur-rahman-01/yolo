from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

img = cv2.imread("C:/Users/Dell/OneDrive/Pictures/Camera Roll/WIN_20251030_01_47_19_Pro.jpg")
 # Add any image in your folder
results = model(img)

annotated = results[0].plot()

cv2.imshow("YOLO image result", annotated)
cv2.waitKey(0)
print("IMAGE:", img)
print("IMAGE:", img)

