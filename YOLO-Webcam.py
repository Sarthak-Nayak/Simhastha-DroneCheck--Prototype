from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)
# cap = cv2.VideoCapture("../Video/bikes.mp4")  # For video

model = YOLO("../Yolo - Weights/yolov8n.pt")

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.3:
                # Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                # Show label
                cvzone.putTextRect(
                    img,
                    f'Person {conf:.2f}',
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=1
                )

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

