from ultralytics import YOLO
import cv2
import cvzone
import math

# --------- Config ---------
VIDEO_SOURCE = 0   # webcam ya CCTV/Video path
PERSON_LIMIT = 5   # Example: Area me max persons allowed

# Define fixed area (rectangle here: x1,y1, x2,y2)
AREA = (100, 100, 500, 500)   # change as per your region

# --------- Load Model ---------
cap = cv2.VideoCapture(VIDEO_SOURCE)
model = YOLO("../Yolo - Weights/yolov8n.pt")

classNames = model.names  # YOLO ke class names (includes "person")

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    person_count = 0

    # Draw ROI rectangle
    x1, y1, x2, y2 = AREA
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if classNames[cls] == "person":
                # Person box
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                w, h = px2 - px1, py2 - py1
                cx, cy = px1 + w // 2, py1 + h // 2  # center point

                # Draw person box
                cvzone.cornerRect(img, (px1, py1, w, h))
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                # Check if inside area
                if x1 < cx < x2 and y1 < cy < y2:
                    person_count += 1
                    cv2.putText(img, "Inside", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)

    # Show person count
    cvzone.putTextRect(img, f'Persons in Area: {person_count}', (20, 50), scale=2, thickness=2, colorR=(0, 0, 255))

    # Alert
    if person_count > PERSON_LIMIT:
        cvzone.putTextRect(img, "⚠ ALERT: Area Overcrowded!", (20, 100), scale=2, thickness=3, colorR=(0, 0, 255))

    cv2.imshow("Monitoring", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
