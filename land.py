import cv2
import numpy as np

# --------- Step 1: CCTV stream ya video file ---------
# Example: replace with your CCTV RTSP URL
# cctv_url = "rtsp://username:password@192.168.1.10:554/stream1"
cctv_url = r"C:\Users\panka\OneDrive\Desktop\pratibha\python\land.mp4"  # testing ke liye ek video file

cap = cv2.VideoCapture(cctv_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# --------- Step 2: Reference empty land image ---------
bg = cv2.imread(r"C:\Users\panka\OneDrive\Desktop\pratibha\python\empty-land.webp")  # khali jagah ki photo
if bg is None:
    print("Error: Empty land image not found!")
    exit()

# Resize background same as video frames
bg = cv2.resize(bg, (640, 480))
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg_gray = cv2.GaussianBlur(bg_gray, (21,21), 0)

# --------- Step 3: Process frames ---------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream not available or video finished")
        break

    frame = cv2.resize(frame, (640, 480))

    # Current frame gray + blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    # --------- Step 4: Difference with background ---------
    diff = cv2.absdiff(bg_gray, gray)
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Clean noise
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    # --------- Step 5: Detect big changes ---------
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 5000:  # threshold area (bada object)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(frame, "⚠️ Unauthorized Structure!", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
            if not ret:  
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # video restart  
                continue  

            if not ret:
                print("No frame received... restarting video")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # video loop karega
                continue

    # --------- Step 6: Show results ---------
    cv2.imshow("Land Monitoring", frame)
    cv2.imshow("Changes Detected", thresh)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
