import cv2
from math import cos, sin
import numpy as np

def draw_labeled_crosshair(img, point, label):
    # Draw a crosshair at the given (x, y) coordinate
    cv2.drawMarker(img, point, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    # Add a text label with the specified text at the given point
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_origin = (point[0] - text_size[0] // 2, point[1] + text_size[1])
    cv2.putText(img, label, text_origin, font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    return img


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img