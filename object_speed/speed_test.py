# import the necessary packages
#from __future__ import print_function
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=64)

# load the image and convert it to grayscale
image = cv2.imread("out1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
mask = cv2.inRange(hsv, greenLower, greenUpper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# find contours in the mask and initialize the current
# (x, y) center of the ball
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[-2]
center = None

#months = (('January',1),('February','2'),('March','3'),'April','May','June', 'July','August','September','October','November','  December')
months = ['January','February','March','April','May','June', 'July','August','September','October','November','  December']
print(type(months), months)

# only proceed if at least one contour was found
if len(cnts) > 0:
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
    #print(cnts)
    print("\n\n\n")
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    # only proceed if the radius meets a minimum size
    if radius > 10:
        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        print(x,y,radius,M)
        cv2.circle(image, (int(x), int(y)), int(radius),
                   (0, 255, 255), 2)
        cv2.circle(image, center, 5, (0, 0, 255), -1)


# show the frame to our screen
cv2.imshow("Original", image)
#cv2.imshow("Gray", gray)
#cv2.imshow("HSV", hsv)
cv2.waitKey(0)
