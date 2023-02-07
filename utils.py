import cv2
import ctypes
from enum import Enum, auto

import numpy as np

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


class Modes(Enum):
    INIT = auto()  # initializing software
    SEARCHING_TRUCK = auto()  # searching id 0
    SEARCHING_GROUP_CAR = auto()  # truck found, searching pem group car
    READY = auto()  # waiting to start the measuring phase, doing small overview with values, no averaging over time
    WAITING_PHASE = auto()  # phase to wait to measuring phase -> 30s
    MEASURING_PHASE = auto()  # measuring phase 30 -> averaging over time -> after going to pem_overlay


def aruco_display(corners, ids, rejected, image):
    """
    Display the detected ArUco markers on the image:

    :param corners: list of detected ArUco markers corners
    :param ids: list of detected ArUco markers IDs
    :param rejected: list of rejected ArUco markers corners
    :param image: image to display the ArUco markers on
    :return: None
    """
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

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))
        # show the output image
    return image


def get_screen_dimensions():
    ## get Screen Size
    user32 = ctypes.windll.user32
    screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    return screen_width, screen_height

def resize_frame(frame, screen_width, screen_height):
    """
    Resizes the given frame to fit within the specified screen dimensions while maintaining the
    aspect ratio of the original frame.

    Args:
        frame: The frame to be resized (an image).
        screen_width: The width of the screen, in pixels.
        screen_height: The height of the screen, in pixels.

    Returns:
        The resized frame.
    """

    # Get the height and width of the frame
    frame_height, frame_width, _ = frame.shape

    # Calculate the scaling factors for the frame based on the screen dimensions
    scale_width = float(screen_width) / float(frame_width)
    scale_height = float(screen_height) / float(frame_height)

    # Determine which scaling factor to use (the smaller one, to ensure the frame fits within the screen)
    if scale_height > scale_width:
        img_scale = scale_width
    else:
        img_scale = scale_height

    # Calculate the new dimensions of the frame
    new_x, new_y = frame.shape[1] * img_scale, frame.shape[0] * img_scale

    # Resize the frame using cubic interpolation
    frame = cv2.resize(frame, (int(new_x), int(new_y)), interpolation=cv2.INTER_CUBIC)

    # Return the resized frame
    return frame


def get_marker_positions(t_vec):
    """

    """
    if t_vec is not None:

        if len(t_vec) == 3:
            x = t_vec[0]
            y = t_vec[1]
            z = t_vec[2]
            return x, y, z

def draw_axis(frame, matrix_coefficients, distortion_coefficients, r_vec, t_vec, num = 0.01):
    cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, r_vec, t_vec, num)


def convert_byte_array_to_string(byte_array: bytearray) -> str:
    """Converts a byte array to a readable string.

    :param byte_array: bytearray or bytes to convert
    :returns: converted string
    :rtype: str

    """
    t = type(byte_array)

    try:
        assert (t is bytearray or t is bytes)

    except AssertionError as msg:
        print("Input: " + str(byte_array) + " is wrong type: " + str(t) + "..." + str(msg))
        return ""

    try:
        string_input = byte_array.decode("utf-8")
    except UnicodeDecodeError:
        string_input = ""
        print("Error while decoding received serial input: {}!".format(byte_array))

    return string_input


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


def draw_cross(frame, center_x, center_y, size=6):

    # Drawing cross on the webcam feed
    center_x = int(center_x)
    center_y = int(center_y)
    try:
        cv2.line(frame, (center_x - int(size/2), center_y), (center_x + int(size/2), center_y), (0, 0, 255), 1)
        cv2.line(frame, (center_x, center_y - int(size/2)), (center_x, center_y + int(size/2)), (0, 0, 255), 1)
    except:
        return frame
    return frame
