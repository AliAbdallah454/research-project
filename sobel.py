import cv2
import numpy as np

img = cv2.imread("./data/ToyData/Session1/Participant1/img233.png", cv2.IMREAD_COLOR)

if img is None:
    print("Image cannot be read")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 1.5)

gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

mag = cv2.magnitude(gx, gy)
mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow("gradient magnitude", mag_u8)
cv2.waitKey(0)