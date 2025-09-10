from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import threading

main = Flask(__name__, template_folder='template')

# Load YOLOv8 model
model = YOLO("../Yolo - Weights/yolov8n.pt")

cap = None
running = False
alert_flag = False

def detect_people():
    global cap, running, alert_flag
    while running and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=0.3, imgsz=480, verbose=False)
        person_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # person
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"People: {person_count}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if person_count > 15:
            alert_flag = True
            cv2.putText(frame, "ALERT: Crowd Detected!", (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            alert_flag = False

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/start')
def start():
    global cap, running
    if not running:
        cap = cv2.VideoCapture("people5.mp4")  # or 0 for webcam
        running = True
    return "Started"

@main.route('/stop')
def stop():
    global cap, running
    running = False
    if cap:
        cap.release()
    return "Stopped"

@main.route('/video')
def video():
    return Response(detect_people(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/alert')
def alert():
    return jsonify({"alert": alert_flag})

if __name__ == '__main__':
    main.run(debug=True)
