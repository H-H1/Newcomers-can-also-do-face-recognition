'''①'''
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
# cnsruct the arumet parse and parse the arguments
ap = argparse. ArgumentParser()
ap.add_argument("-v","--video",help="path to the (optional) video file")
ap.add_argument("-b","--buffer",type= int, default=64,help="max buffer size")
args = vars(ap.parse_args())
# 井define the lower and upper boundaries of the "yel low objeet
# (or "al'’intheHSVcolor space, then initalize the
# # list of tracked points
colorLower = (24, 100, 100)
colorUpper= (44, 255,255)
pts=deque(maxlen=args["buffer"])
#if a video path was not spplied,
# 井to the webcam
# grab the reference
if not args.get("video", False):
    camera=cv2.VideoCapture(0)
# 善therie, 8rab a reterene to the
else:
    camera = cv2. VideoCapture(args["video"])
# keep looping
while True:
# grab the current frame
    (grabbed, frame) = camera. read()

    if args.get("video") and not grabbed:
        break
    frame=imutils.resize(frame,width=600)
    frame=imutils.rotate(frame,angle=180)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    mask=cv2.inRange(hsv,colorLower,colorUpper)
    mask=cv2.erode(mask,None,iterations=2)
    mask=cv2.dilate(mask,None,iterations=2)
    cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    center=None
    if len(cnts)>0:
        c=max(cnts,key=cv2.contourArea)
        ((x,y),radius)=cv2.minEnclosingCircle(c)
        M=cv2.moments(c)
        center=(int(M["m10"])/M["m00"]),int(M["m01"]/M["m00"])
        if radius>10:
            cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
            cv2.circle(frame,center,5,(0,0,255),-1)
    pts.appendleft(center)
    for i in range(1,len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue

        thickness=int(np.sqrt(args["buffer"]/float(i+1))*2.5)
        cv2.line(frame,pts[i-1],pts[i],(0,0,255),thickness)
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break
camera.release()
cv2.destroyAllWindows()

'''②'''

