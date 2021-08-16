#
# Based on https://pysource.com/2019/01/10/eye-blinking-detection-gaze-controlled-keyboard-with-python-and-opencv-p-2/
#

import cv2
import numpy as np
import dlib
from math import hypot
from math import sqrt
import os       # for the speaker

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

# compute formula to understand if eye is closed or not --> more precise formula
def compute_EAR(eye_points, facial_landmarks):
	p1 = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
	p4 = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
	p2 = (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y)
	p3 = (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y)
	p5 = (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y)
	p6 = (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)
	EAR = (norm_of_difference(p2,p6) + norm_of_difference(p3,p5)) / (2 * norm_of_difference(p1,p4))
	return EAR

# utility function
def norm_of_difference(x, y):
	norm = 0
	x_part = x[0] - y[0]
	y_part = x[1] - y[1]	
	norm += sqrt(x_part**2 + y_part**2)
	return norm

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        #######
        # even if the ouput value is not needed, it is needed to draw the eyes on the camera
        #######

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

        #######
        # Old version
        #######
        # print("left", left_eye_ratio)
        # print("right", right_eye_ratio)
        # blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        # print("ratio", blinking_ratio, "\n")

        # os.system --> speaks out the specified string

        # if blinking_ratio > 6:
        #     os.system('spd-say "wake up"')  # speaker
        # elif left_eye_ratio > 6.5:
        #     os.system('spd-say "right"')  # speaker
        # elif right_eye_ratio > 6.5:
        #     os.system('spd-say "left"')  # speaker

        # if blinking_ratio > 5.7:
            # cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
            # os.system('spd-say "wake up"')  # speaker

        ######
        # New part: just compute the EAR --> not sure how to react to it
        ######
        EAR_left_eye = compute_EAR([36, 37, 38, 39, 40, 41], landmarks)
        EAR_right_eye = compute_EAR([42, 43, 44, 45, 46, 47], landmarks)

        print("LEFT", EAR_left_eye)
        print("RIGHT", EAR_right_eye)
        print("")




    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()