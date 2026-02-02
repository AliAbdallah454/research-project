import cv2
import numpy as np

def detect_red_circle(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 40, 40], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)

    lower2 = np.array([170, 40, 40], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No reddish region found (adjust HSV thresholds).")
    else:

        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) < 500:
            print("Red region too small (likely noise). Try lowering/raising HSV thresholds.")
        else:
            (x, y), r = cv2.minEnclosingCircle(c)
            x, y, r = int(round(x)), int(round(y)), int(round(r))
            return mask, x, y, r
        return None, 0, 0, 0
    return None, 0, 0, 0

if __name__ == "__main__":
        
    img = cv2.imread("./data/Session1/Participant1/img233.png", cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError("Image not found ...")

    output = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 40, 40], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)

    lower2 = np.array([170, 40, 40], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No reddish region found (adjust HSV thresholds).")
    else:

        # Pick the largest red region (most likely the disk)
        c = max(contours, key=cv2.contourArea)

        # Optional: ignore tiny blobs
        if cv2.contourArea(c) < 500:
            print("Red region too small (likely noise). Try lowering/raising HSV thresholds.")
        else:
            # 5) Fit circle around that region
            (x, y), r = cv2.minEnclosingCircle(c)
            x, y, r = int(round(x)), int(round(y)), int(round(r))

            # Draw result
            cv2.circle(output, (x, y), r, (0, 255, 0), 3)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

    cv2.imshow("detected reddish circle", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()