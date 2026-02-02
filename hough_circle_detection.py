import cv2
import numpy as np

img = cv2.imread("./data/ToyData/Session1/Participant1/img233.png", cv2.IMREAD_COLOR)

if img is None:
    print("Image cannot be read")

output = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=50,
    param1=80,
    param2=30,
    minRadius=30,
    maxRadius=600
)

h, w = gray.shape[:2]
cx, cy = w // 2, h // 2

if circles is not None:
    circles = np.int32(np.around(circles))
    
    closest_to_center = min(circles[0], key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)
    furthest_to_center = max(circles[0], key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)
    smalles_radius = min(circles[0], key=lambda c: c[2])

    t = min(circles[0], key=lambda c: ((c[0]-cx)**2 + (c[1]-cy)**2, -c[2]))

    x, y, r = t

    cv2.circle(output, (x, y), r, (0, 255, 0), 5)
    cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

else:
    print("No circles detected ...")

cv2.imshow('Detected Circle', output)
cv2.waitKey(0)
cv2.destroyAllWindows()