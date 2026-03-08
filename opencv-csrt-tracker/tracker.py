import cv2
import time

# Select The camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

tracker = None
bbox = None
tracking = False
prev_time = 0

while True:

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    if tracking:

        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Tracking",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "Tracking Lost",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # calculate FPS to measure latency
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # steps to guide the user
    cv2.putText(
        frame,
        "Press S: Select Object | Q: Quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Real-Time Object Tracker", frame)

    key = cv2.waitKey(1) & 0xFF

    # test the object
    if key == ord("s"):

        bbox = cv2.selectROI(
            "Real-Time Object Tracker",
            frame,
            fromCenter=False,
            showCrosshair=True
        )

        # Validate the choice
        if bbox[2] > 0 and bbox[3] > 0:

            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)
            tracking = True

    # exit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()