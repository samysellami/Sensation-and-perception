import cv2
import numpy as np


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


image = cv2.imread('image.bmp')
depth = cv2.imread('depth.bmp')


blurred = cv2.pyrMeanShiftFiltering(image, 21, 51)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

_, contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
box = None
for contour in contours:

    # get rectangle bounding contour

    [x, y, w, h] = cv2.boundingRect(contour)
    # discard areas that are too large
    if h == 123 and w == 82:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

(bl, tl, tr, br) = box
(tltrX, tltrY) = midpoint(tl, tr)
(blbrX, blbrY) = midpoint(bl, br)
(tlblX, tlblY) = midpoint(tl, bl)
(trbrX, trbrY) = midpoint(tr, br)

cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

cv2.circle(image, (int(tltrX), int(tltrY)), 3, (255, 0, 0), -1)
cv2.circle(image, (int(blbrX), int(blbrY)), 3, (255, 0, 0), -1)
cv2.circle(image, (int(tlblX), int(tlblY)), 3, (255, 0, 0), -1)
cv2.circle(image, (int(trbrX), int(trbrY)), 3, (255, 0, 0), -1)

centerX = (tltrX+blbrX)/2
centerY = (tlblY+trbrY)/2
print("Center:", centerX, centerY)


cv2.circle(image, (int(centerX), int(centerY)), 3, (255, 0, 0), -1)
centerXdepth = centerX*(512/1280)
centerYdepth = centerY*(424/720)

print("Center in depth:", centerYdepth, centerXdepth)
cv2.circle(depth, (int(centerYdepth), int(centerXdepth)), 3, (0, 0, 255), -1)

print(depth[int(centerYdepth), int(centerXdepth)])
cv2.imshow("", depth)
cv2.waitKey()
cv2.imshow("", image)
cv2.waitKey()
z = 32*4
