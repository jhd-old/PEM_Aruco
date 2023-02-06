import cv2
import time
import serial
#import vlc
import datetime
import numpy as np
from typing import Optional
from utils import ARUCO_DICT, get_screen_dimensions, resize_frame, Modes, get_marker_positions, draw_axis, convert_byte_array_to_string
# import required libraries
from vidgear.gears import CamGear


# define suitable tweak parameters for your stream.
options = {
    "CAP_PROP_FRAME_WIDTH": 320, # resolution 320x240
    "CAP_PROP_FRAME_HEIGHT": 240,
    "CAP_PROP_FPS": 30, # framerate 60fps
}

WINDOW_NAME = 'PEM1 WS22/23'

# für CV2 4.7
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# CV2 4.6
# arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
# arucoParams = cv2.aruco.DetectorParameters_create()

#cap = cv2.VideoCapture(0)
stream = CamGear(source=0, logging=True, **options).start()
# Full screen mode
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.startWindowThread()

# is needed to set the window to full screen
def on_change(value):
    pass

# could be used to set the distance
# cv2.createTrackbar('distance', WINDOW_NAME, 25, 100, on_change)


try: 
    while(True):
        #ret, frame = cap.read()
        ret,frame = True, stream.read()
        
        # check for frame if Nonetype
        if frame is None:
            break

        corners, ids, rejected = detector.detectMarkers(frame)
        print("frame size: " + str(frame.size))
        #cv2.waitKey(1) 

        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # draw the bounding box of the ArUCo detection
                cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                # draw the ArUco marker ID on the image
                #text = "Gruppe: " + str(markerID) + " \n" + "Abweichung x: " + str(cX) + "px" + " \n" + "Abweichung y: " + str(cY) + "px"

                # y0, dy = 50, 50
                # for i, line in enumerate(text.split('\n')):
                #     y = y0 + i*40
                #     cv2.putText(frame, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)


        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except: 
    #cap.release()
    stream.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(5)
finally:
    #cap.release()
    stream.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(5)