import cv2
import numpy as np
from red_circle_detection import detect_red_circle

# cap = cv2.VideoCapture("./data/Session1/Participant1/video.mp4")
cap = cv2.VideoCapture("./data/ToyData/Session3/Participant18/video.mp4")

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    mask, x, y, r = detect_red_circle(frame)

    cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
    cv2.circle(frame, (x, y), 2, (0, 0, 255), 2)

    cv2.imshow("Video", frame)
    if mask is not None:
        cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()