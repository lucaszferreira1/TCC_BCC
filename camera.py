from ultralytics import YOLO
import cv2

K_ESC = 27

WIDTH = 640
HEIGHT = 480
FPS = 15

# Load a model
model = YOLO("yolo11n.yaml")
model = YOLO("yolo11n.pt")
model = YOLO("yolo11n.yaml").load("best.pt")

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)
    
    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{box.cls[0]} {box.conf[0]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("YOLO Inference", frame)
    if cv2.waitKey(1) & 0xFF == K_ESC:
        break

cap.release()
cv2.destroyAllWindows()