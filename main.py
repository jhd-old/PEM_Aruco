import cv2
import time

import serial
import vlc
import datetime
import numpy as np
from typing import Optional
from utils import ARUCO_DICT, get_screen_dimensions, resize_frame, Modes, get_marker_positions, draw_axis, \
    convert_byte_array_to_string

mode = Modes.INIT
type = "DICT_4X4_50"
WINDOW_NAME = 'PEM1 WS22/23'
marker_size_in_CM = 15.9
results = dict()
waiting, measuring = False
set_point = 25
reset = False
waiting_time = 30
waiting_ref_time = None
measuring_time = 30
measuring_ref_time = None
font = cv2.FONT_HERSHEY_SIMPLEX
overlay = None

music = vlc.MediaPlayer("sources/icecream_truck.mp3")

# to retrieve the energy data maybe?
ser = serial.Serial(
    # Serial Port to read the data from
    port='COM3',

    # Rate at which the information is shared to the communication channel
    baudrate=9600,

    # Applying Parity Checking (none in this case)
    parity=serial.PARITY_NONE,

    # Pattern of Bits to be read
    stopbits=serial.STOPBITS_ONE,

    # Total number of bits to be read
    bytesize=serial.EIGHTBITS,

    # Number of serial commands to accept before timing out
    timeout=1
)

# initialize video capture object to read video from external webcam
video_capture = cv2.VideoCapture(1)
# if there is no external camera then take the built-in camera
if not video_capture.read()[0]:
    video_capture = cv2.VideoCapture(0)

time.sleep(2.0)


# is needed to set the window to full screen
def on_change(value):
    pass


# Full screen mode
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# could be used to set the distance
cv2.createTrackbar('distance', WINDOW_NAME, set_point, 100, on_change)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])
arucoParams = cv2.aruco.DetectorParameters_create()

# if we want to calibrate the camera to retrieve measurements in m.
m = np.load("calib/m")
d = np.load("calib/d")

# main loop, magic happens here
while video_capture.isOpened():
    screen_width, screen_height = get_screen_dimensions()

    set_point = cv2.getTrackbarPos('distance', WINDOW_NAME)

    # use an overlay to draw on top of the video
    if overlay is None:
        # Create a blank black image
        overlay = np.zeros((screen_width, screen_height, 3), np.uint8)
        # Fill image with red color(set each pixel to red)
        if not ser.is_open:
            overlay[:] = (0, 0, 255)
    elif overlay.shape[0] is not screen_height or overlay.shape[1] is not screen_width:
        overlay = np.zeros((screen_width, screen_height, 3), np.uint8)
        # Fill image with red color(set each pixel to red)
        if not ser.is_open:
            overlay[:] = (0, 0, 255)

    # retrieve frame
    ret, frame = video_capture.read()

    g_id: Optional[int] = None  # group id, 1-6, aruco marker id of the group. None if not found yet
    t_id: int = 0  # truck id, 0, aruco marker id of the truck

    # break if we dont receive an image
    if ret is False:
        break
    else:
        # resize frame to fit the screen
        frame = resize_frame(frame, screen_width, screen_height)

        # TODO check if we need to rotate the image
        # image = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
        # image = cv2.rotate(src, cv2.ROTATE_180)
        # image = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)

        ###########
        # DETECTION
        ###########

        # detect markers in image
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

        # any id found?
        if ids is not None:
            if len(ids) > 0:
                if len(ids) <= 2:

                    # truck id found?
                    if t_id in ids:

                        # check if any group id found
                        if any(x in ids for x in range(1, 7)):
                            # we are ready to measure
                            mode = Modes.READY
                        else:
                            # truck found, but no group id. Waiting for group id to be found
                            mode = Modes.SEARCHING_GROUP_CAR
                else:
                    pass
                    # TODO: error
            else:
                # not even id=0 (truck) found
                mode = Modes.SEARCHING_TRUCK
        else:
            # not even id=0 (truck) found
            # TODO: check what todo here: Text to adjust camera
            mode = Modes.SEARCHING_TRUCK
            pass

        ##############
        # CALCULATIONS
        ##############

        # if we are ready to measure do calculations
        if mode == Modes.READY:

            # get group id
            g_id = [num for num in ids if not t_id]

            # reset groups last results
            if reset:
                results.pop(g_id, None)

            # retrieve poses
            # 1. get the rotation and translation vectors of the truck
            truck_r_vec, truck_t_vec, trucK_markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[t_id], 0.02, m,
                                                                                               d)
            # 2. get the rotation and translation vectors of the car
            car_r_vec, car_t_vec, car_markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[g_id], 0.02,
                                                                                         m, d)

            # Convert vectors to positions
            truck_x, truck_y, truck_z = get_marker_positions(truck_t_vec)
            car_x, car_y, car_z = get_marker_positions(car_t_vec)

            # Add a overlay with the results to the current frame
            if not measuring:
                # apply overlay
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha,
                                0, frame)

                org = (50, 50)
                # Blue color in BGR
                color = (255, 0, 0)

                # TODO: current x and y on top
                frame = cv2.putText(frame, "X: %  Y: %".format(car_x, car_y), org, font,
                                    1, color, 2, cv2.LINE_AA)

                # TODO: 2. Groups overlay
                distance_y = 20
                some_missing = False
                for i in range(1, 7):
                    group_results = results.get(g_id, {})
                    if group_results:

                        x_diff = group_results.get("x_diff", "/")
                        y_diff = group_results.get("y_diff", "/")
                        x_var = group_results.get("var_x", "/")
                        y_var = group_results.get("var_y", "/")
                        e = group_results.get("e", "/")

                        if x_diff != "/":
                            x_diff_average = np.mean(x_diff)
                        if y_diff != "/":
                            y_diff_average = np.mean(y_diff)

                        frame = cv2.putText(frame, "Gruppe %: ⌀X =% | VarX=% | ⌀Y: % | VarY=% | ∑E=%".format(i, ), org,
                                            font,
                                            1, color, 2, cv2.LINE_AA)
                        org_y = org[1] + distance_y
                        org = (org[0], org_y)
                    else:
                        # some results are still missing
                        some_missing = True

                if some_missing:
                    # TODO: 1. Press 's' to start
                    frame = cv2.putText(frame, "Press 's' to start!", org, font,
                                        1, color, 2, cv2.LINE_AA)

                if waiting:

                    if waiting_ref_time is None:
                        waiting_ref_time = time.time()

                    time_diff_s = time.time() - waiting_time
                    time_last = waiting_time - time_diff_s

                    # TODO: additionally show countdown
                    org = (50, 50)
                    # Blue color in BGR
                    color = (255, 0, 0)

                    # TODO: current x and y on top
                    frame = cv2.putText(frame, str(time_last), org, font,
                                        3, color, 2, cv2.LINE_AA)

                    # draw
                    draw_axis(frame, m, d, truck_r_vec, truck_t_vec, 0.01)
                    draw_axis(frame, m, d, car_r_vec, car_t_vec, 0.01)

                    if time_last >= 0:
                        waiting_ref_time = None
                        measuring = True
                        waiting = False
            else:

                if not music.is_playing():
                    music.play()

                if measuring_ref_time is None:
                    measuring_ref_time = time.time()

                if ser.isOpen():
                    ser.write(b'get')

                    try:
                        received_line = ser.readline()
                    except:
                        received_line = ""

                    if len(received_line) > 0:
                        string_line = convert_byte_array_to_string(received_line)

                        try:
                            e = float(string_line)
                        except ValueError:
                            e = None
                    else:
                        e = None
                else:
                    e = None

                time_diff_s = time.time() - measuring_time
                time_last = measuring_time - time_diff_s

                # get data
                group_results = results.get(g_id, {})
                groups_results_x = group_results.get("x", [])
                groups_results_y = group_results.get("y", [])
                groups_results_x_diff = group_results.get("x_diff", [])
                groups_results_y_diff = group_results.get("y_diff", [])
                groups_results_e = group_results.get("e", [])

                # calculate data
                diff_x = truck_x - car_x
                diff_y = truck_y - car_y

                diff_to_set_point_y = set_point - diff_y

                groups_results_x.append(car_x)
                groups_results_y.append(car_y)
                groups_results_y_diff.append(diff_to_set_point_y)
                groups_results_x_diff.append(diff_x)

                if e is not None:
                    groups_results_e.append(e)

                var_x = np.var(groups_results_x)
                var_y = np.var(groups_results_y)

                # save data
                group_results.update({"x": groups_results_x})
                group_results.update({"y": groups_results_y})
                group_results.update({"x_diff": groups_results_x_diff})
                group_results.update({"y_diff": groups_results_y_diff})
                group_results.update({"var_x": var_x})
                group_results.update({"var_y": var_y})

                # draw
                draw_axis(frame, m, d, truck_r_vec, truck_t_vec, 0.01)
                draw_axis(frame, m, d, car_r_vec, car_t_vec, 0.01)

                org = (50, 50)
                # Blue color in BGR
                color = (255, 0, 0)

                frame = cv2.putText(frame, "Test", org, font,
                                    1, color, 2, cv2.LINE_AA)

                # TODO: additionally show countdown
                org = (50, 50)
                # Blue color in BGR
                color = (255, 0, 0)

                frame = cv2.putText(frame, str(time_last), org, font,
                                    3, color, 2, cv2.LINE_AA)

                if time_last <= 0:
                    measuring_ref_time = None
                    measuring = False

                    year = datetime.date.today().year  # the current year
                    month = datetime.date.today().month  # the current month
                    day = datetime.date.today().day  # the current day
                    hour = datetime.datetime.now().hour  # the current hour
                    minute = datetime.datetime.now().minute  # the current minute
                    seconds = datetime.datetime.now().second  # the current second

                    file_name = "data/" + "%-%-%_%:%:%.npy".format(year, month, day, hour, minute, seconds)
                    np.save(file_name, results)

                    music.pause()

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        if not measuring:
            waiting = True
            reset = True

cv2.destroyAllWindows()
video_capture.release()
