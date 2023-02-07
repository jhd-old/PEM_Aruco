import time

import cv2
import numpy as np

from utils import overlay_transparent, draw_cross

try:
    import vlc

    play_music = True
except ImportError:
    print("vlc not found")
    play_music = False
    vlc = None

try:
    pem_overlay = cv2.imread('pem_small.png')
except FileNotFoundError:
    print("pem.png not found")
    pem_overlay = None

WINDOW_NAME = 'PEM1 WS22/23'

# für CV2 4.7
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(0)
# Full screen mode
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.startWindowThread()


measuring = False
group_id = None
markerID = None
show_results = False
measurement_time = 30
wanted_distance_in_cm = 25
distance_offset_in_cm = 1.5
wanted_distance_in_px = None

# TODO: check x offset!
x_offset = 0

x_dev = None
y_dev = None

# TODO: think about a penalty system
penalty = 0

# TODO: check if we need to flip the frame
# TODO: check distance offset
# TODO: check renaming x and y in text to Spur and Abstand?

# initialize the music player, if vlc is installed
if play_music:
    music = vlc.MediaPlayer("sources/icecream_truck.mp3")
else:
    music = None

overlay = None

try:

    text = None
    pixel_to_cm_ratio = None

    # truck marker center point
    tX, tY = None, None

    while True:
        # Capture current frame
        ret, frame = cap.read()

        if not ret or frame is None:
            time.sleep(1)
            continue

        # get the frame size
        h, w = frame.shape[:2]

        if overlay is None:
            img_white = np.ones((h, w, 3), np.uint8) * 255

            pem_overlay_h = pem_overlay.shape[0]
            overlay = overlay_transparent(img_white, pem_overlay, 10, h - 10 - pem_overlay_h)

        # get the mid point of the frame
        mid_point = (w / 2, h / 2)

        # detect markers on grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect ArUco markers in the input frame
        corners, ids, rejected = detector.detectMarkers(frame_gray)

        # check to see if the ArUco detector was able to detect at least one
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)

                # truck marker is detected
                if markerID == 0:
                    detect_truck = True

                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    # truck marker center x and y
                    tX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    tY = int((topLeft[1] + bottomRight[1]) / 2.0)

                    int_corners = np.int0(markerCorner)
                    cv2.polylines(frame, int_corners, True, (255, 255, 0), 1)

                    if len(corners) == 1:
                        # no car detected
                        text = None
                        x_dev = None
                        y_dev = None

                    # car marker is detected
                elif 1 <= markerID < 7:
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners

                    # compute and draw the center (x, y)-coordinates of the ArUco
                    # marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                    frame = draw_cross(frame, cX, cY, 4)

                    # Aruco Perimeter
                    aruco_perimeter = cv2.arcLength(markerCorner[0], True)
                    # Pixel to cm ratio
                    pixel_to_cm_ratio = aruco_perimeter / (3 * 3)

                    wanted_distance_in_px = (wanted_distance_in_cm + distance_offset_in_cm) * pixel_to_cm_ratio

                    if tX is not None and tY is not None and wanted_distance_in_px is not None:
                        x_dev = np.round((tX + x_offset - cX) / pixel_to_cm_ratio, 1)
                        y_dev = np.round((tY + wanted_distance_in_px - cY) / pixel_to_cm_ratio, 1)

                    else:
                        x_dev = np.round((mid_point[0] - cX) * pixel_to_cm_ratio, 1)
                        y_dev = np.round((mid_point[1] - cY) * pixel_to_cm_ratio, 1)

                    # draw the ArUco marker ID on the image
                    text = "Gruppe: " + str(markerID) + " \n" + "Abweichung x: " + str(
                        x_dev) + "cm" + " \n" + "Abweichung y: " + str(y_dev) + " cm"

                    int_corners = np.int0(markerCorner)
                    cv2.polylines(frame, int_corners, True, (0, 255, 0), 1)

                    if markerID > 0:
                        group_id = markerID
                else:
                    text = None
                    if not show_results:
                        group_id = None

        else:
            cX = None
            cY = None
            text = None

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            if not measuring:
                measuring = True
                tic = time.time()
                results = {"x": [], "y": []}
            else:
                measuring = False
                break
        elif key == ord("r"):
            show_results = False
            measuring = False
            results = {"x": [], "y": []}
            group_id = None
            text = None
            x_dev = None
            y_dev = None
            pixel_to_cm_ratio = None
            wanted_distance_in_px = None

        if measuring:

            if music is not None and play_music:
                if not music.is_playing():
                    music.play()

            # add penalty if no marker is detected
            if x_dev is None:
                x_dev = penalty
            if y_dev is None:
                y_dev = penalty

            results["x"].append(x_dev)
            results["y"].append(y_dev)

            dif = int(time.time() - tic)
            cv2.putText(frame, str(dif) + "/" + str(measurement_time) + "s", (w - 110, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 1, cv2.LINE_AA)

            if dif > measurement_time:
                show_results = True
                measuring = False

                if music is not None:
                    music.stop()

        if show_results:

            img_results = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0.0)

            mean_x = np.round(np.mean(results["x"]), 2)
            mean_y = np.round(np.mean(results["y"]), 2)

            std_x = np.round(np.std(results["x"]), 2)
            std_y = np.round(np.std(results["y"]), 2)

            text = "Gruppe: " + str(group_id) + " \n" + "Abweichung x: " + str(
                mean_x) + "cm" + " \n" + "Abweichung y: " + str(
                mean_y) + " cm" + " \n" + "Standardabweichung x: " + str(
                std_x) + "cm" + " \n" + "Standardabweichung y: " + str(std_y) + " cm"

            y0, dy = 30, 30

            for i, line in enumerate(text.split('\n')):
                y = y0 + i * 35
                cv2.putText(img_results, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, img_results)
        else:

            if len(corners) > 0:
                cross_size = 20
                if tX is not None and tY is not None and wanted_distance_in_px is not None:

                    frame = draw_cross(frame, tX + x_offset, tY + wanted_distance_in_px, size=cross_size)
                else:
                    frame = draw_cross(frame, mid_point[0], mid_point[1], size=cross_size)

            if not measuring:
                frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0.0)

            y0, dy = 30, 30

            if text is not None:
                for i, line in enumerate(text.split('\n')):
                    y = y0 + i * 40
                    cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)


except Exception as e:
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(5)
    print(e)
finally:
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(5)
