import cv2
import imutils
from utils import get_rect_vertices, get_point_coords, calculate_ear, notify
import dlib
import time
from playsound import playsound
import sys

user_video = cv2.VideoCapture(0)

# user_video.open("http://192.168.0.4:8080/video") # Im using my phone's camera as my laptop webcam quality is too bad

face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor(
    "./shape_predictor_68_face_landmarks.dat")


def get_threshold():
    return float(open("threshold.txt", "r").read())

def set_threshold(t):
    f = open("threshold.txt", "w")
    f.write(str(t))
    f.close()

THRESHOLD = get_threshold()
FRAME_COUNT = 3

COUNTER = 0
BLINK_COUNT = 0

timer = time.time()
index = 0
calibrate = True

while True:

    e, frame = user_video.read()

    try:
        frame = imutils.resize(frame, width=600)
    except AttributeError:
        notify("Heyelth - Error", "Please check your webcam and calibrate again!")
        exit()

    unchanged_frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting Faces
    detected_faces = face_detector(gray)

    for face in detected_faces:

        # Drawing a Rectangle on the detected faces.
        x1, x2, y1, y2 = get_rect_vertices(face)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)

        landmarks = landmarks_predictor(gray, face)

        left_eye_points = []
        for i in range(36, 43):
            points = tuple(get_point_coords(landmarks.part(i)))
            left_eye_points.append(points)
            cv2.circle(frame, points, 1, (255, 0, 255), 1)

        right_eye_points = []
        for i in range(42, 48):
            points = tuple(get_point_coords(landmarks.part(i)))
            right_eye_points.append(points)
            cv2.circle(frame, points, 1, (0, 255, 255), 1)

        leftEyeEAR = calculate_ear(left_eye_points)
        rightEyeEAR = calculate_ear(right_eye_points)

        averageEAR = (leftEyeEAR + rightEyeEAR) / 2

        if averageEAR < THRESHOLD:
            COUNTER += 1
        else:
            if COUNTER >= FRAME_COUNT:
                BLINK_COUNT += 1
            COUNTER = 0

        cv2.putText(frame, f"Total Blinks: {BLINK_COUNT}",
                    (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 150), 2)

        cv2.putText(frame, f"EAR: {str(averageEAR)[:4]}",
                    (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 150, 0), 2)
        
        cv2.putText(frame, f"Threshold: {str(THRESHOLD)[:4]}",
                    (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

        if (calibrate):

            cv2.putText(frame, str(index), (200, 270), cv2.FONT_HERSHEY_DUPLEX, 10, (0, 0, 0), 10)
            

    if time.time() - timer >= 2:
        timer = time.time()
        index += 1
    
    if index == 10 and calibrate:
        print("Blinked", BLINK_COUNT, "times")
        if BLINK_COUNT > 10:
            set_threshold(THRESHOLD + 0.015)

        elif BLINK_COUNT < 9:
            set_threshold(THRESHOLD - 0.01)
        
        if THRESHOLD == get_threshold():
            calibrate = False
            print("Threshold is the same")
            exit()
            
        THRESHOLD = get_threshold()
        calibrate = False
        exit()
        

    cv2.imshow("Debug Feed", frame)
    # cv2.imshow("Unchanged User Feed", unchanged_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


user_video.release()
cv2.destroyAllWindows()
exit()
