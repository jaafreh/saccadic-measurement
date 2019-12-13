#marker or object with a known width(width)is placed
# on distance D from our camera. apparent width in pixels dP. focal length Fl of our camera:
 # Fl = (dP x  D) / W

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import dlib
webcam = cv2.VideoCapture(0)
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
width=45  # width of the marker in mm
D=450  # distance of the marker in mm
# construct the argument parse and parse the arguments
def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and, then initialize the
    # distance colors and reference object
    (cnts, _) = contours.sort_contours(cnts)
    colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
              (255, 0, 255))
    refObj = None

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 500: #or cv2.contourArea(c) > 4000 :
            continue
   # c = max(cnts, key=cv2.contourArea)
    # compute the bounding box of the of the paper region and return it
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    (tl, tr, br, bl) = box
    db = dist.euclidean(tl,tr)

    return db
_, frame = webcam.read()
# show the output image
orig = frame.copy()
m=find_marker(frame)

fl=m*D/width  # fl =930 this basded on my experiment
#  D = (W x F) / P
 # W = (dP x  D) / Fl

#print("number of pixel=",m)
print("focal length =",fl)
cv2.imshow("Image", orig)
cv2.waitKey(0)