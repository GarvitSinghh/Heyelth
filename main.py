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

THRESHOLD = get_threshold()
FRAME_COUNT = 3

COUNTER = 0
BLINK_COUNT = 0

start_time = time.time()

t1 = time.time()
check20 = None

while True:

    e, frame = user_video.read()
    try:
        frame = imutils.resize(frame, width=600)
    except AttributeError:
        notify("Heyelth - Error", "Please check your webcam and run again!")
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

        curr_time = time.time()
        time_elapsed = time.time() - start_time
        if time_elapsed >= 60:
            if BLINK_COUNT < 15:
                playsound("./utils/notif_sound.mp3")
                notify("Heyelth - Dont forget to Blink!!", f"You blinked only {BLINK_COUNT} times this minute! Make sure to blink atleast 20 times!")
                print("", "", "DONT FORGET TO BLINK MORE",
                      f"YOU ONLY BLINKED {BLINK_COUNT} TIMES THIS MINTUE", sep="\n")
            start_time = time.time()
            BLINK_COUNT = 0
            COUNTER = 0

        cv2.putText(frame, f"Total Blinks: {BLINK_COUNT}",
                    (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"EAR: {str(averageEAR)[:4]}",
                    (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 150, 0), 2)

        cv2.putText(frame, f"Time: {int(time_elapsed)}",
                    (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

        
    t2 = time.time()
    if t2 - t1 >= 1200:
        t1 = time.time()
        start_time = time.time()

        playsound("./utils/notif_sound.mp3")
        notify("Heyelth - Take a break!", "Stare at something 20 metres away for 20 seconds!")
        # print("Take a break!", "Stare at something 20 metres away for 20 seconds!")

        check20 = time.time()
    
    if check20:
        if time.time() - check20 >= 20:
            t1 = time.time()
            start_time = time.time()
            playsound("./utils/notif_sound.mp3")
            notify("Heyelth - Continue your work!", "20 seconds have passed. Good luck with your work!!")
            # print("Continue your work!", "20 seconds have passed. You should now continue!")
            check20 = None

    # cv2.imshow("Debug Feed", frame)
    # cv2.imshow("Unchanged User Feed", unchanged_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


user_video.release()
cv2.destroyAllWindows()
