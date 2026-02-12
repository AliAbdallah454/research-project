import cv2
import numpy as np
import argparse

from red_circle_detection import detect_red_circle

def build_args():
    p = argparse.ArgumentParser(description="Run CircleRegressor on one frame.")
    p.add_argument("--session", type=int, required=True, help="Session number, e.g. 2")
    p.add_argument("--participant", type=int, required=True, help="Participant number, e.g. 14")

    return p.parse_args()

args = build_args()

video_path = f"./data/Cornia/Session{args.session}_Light/Participant{args.participant}/video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    mask, c = detect_red_circle(frame)

    h, w, _ = frame.shape
    x = int(c[0] * w)
    y = int(c[1] * h)
    r = int(c[2] * min(w, h))

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