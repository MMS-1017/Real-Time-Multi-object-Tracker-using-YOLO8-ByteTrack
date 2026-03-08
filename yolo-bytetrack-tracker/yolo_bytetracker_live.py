import cv2
import time
from ultralytics import YOLO
import supervision as sv

# Load YOLO model
model = YOLO("yolov8n.pt")  # nano version to test on CPU
tracker = sv.ByteTrack()     # Multi-object tracker

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

selected_ids = set()  # set of IDs for manually selected objects
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_display = frame.copy()
    curr_time = time.time()
    fps = 1/(curr_time-prev_time) if prev_time else 0
    prev_time = curr_time

    # YOLO detection
    results = model(frame, imgsz=640, conf=0.4)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Update ByteTrack
    tracked = tracker.update_with_detections(detections)

    # Draw all tracked objects
    for xyxy, track_id, cls_id in zip(tracked.xyxy, tracked.tracker_id, tracked.class_id):
        x1, y1, x2, y2 = map(int, xyxy)
        if track_id in selected_ids:
            color = (0,255,0)  # green for selected objects
            label = f"Selected {model.names[int(cls_id)]} ID:{track_id}"
        else:
            color = (255,255,0)  # blue for others
            label = f"{model.names[int(cls_id)]} ID:{track_id}"
        cv2.rectangle(frame_display,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame_display,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # Show FPS & instructions
    cv2.putText(frame_display,f"FPS: {int(fps)}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(frame_display,"Press S: Select Object | Q: Quit",(20,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.imshow("YOLO + ByteTrack Tracker", frame_display)
    key = cv2.waitKey(1) & 0xFF

    # Select object live
    if key == ord("s"):
        
        roi_bbox = cv2.selectROI("Select ROI", frame.copy(), False)
        cv2.destroyWindow("Select ROI")

        x, y, w, h = [int(v) for v in roi_bbox]
        cx, cy = x + w//2, y + h//2

        selected_ids.clear()
        for xyxy, track_id in zip(tracked.xyxy, tracked.tracker_id):
            dx1, dy1, dx2, dy2 = map(int, xyxy)
            if dx1 <= cx <= dx2 and dy1 <= cy <= dy2:
                selected_ids.add(track_id)
                print(f"Selected object ID: {track_id}")
                break
        if not selected_ids:
            print("No object detected inside your box. Try again.")

    # Exit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()