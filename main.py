import mediapipe as mp
import pyvirtualcam
import cv2
import numpy as np
import csv
import scipy
from utils import calc_landmark_list, pre_process_landmark
import math
import time

'''Set parameters '''
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

DEBUG = not True

angle_speed = 45
motion_speed = 10
position_limits = [-100, -50]
face = ((180, 50), (325, 200))
face_size = (face[1][0] - face[0][0], face[1][1] - face[0][1])

amplitude = abs(position_limits[1] - position_limits[0]) / 2
k = (position_limits[0] + position_limits[1]) / 2

enot = cv2.imread('images/enot.png', -1)
enot = cv2.resize(enot, (450, 450))

image_template = np.zeros((480, 640, 3), dtype=np.uint8)
side_length = 50
radius = 200

'''____________________________'''

h, w = image_template.shape[:2]
center = (w // 2, h // 2)


cv2.circle(image_template, center, radius, (255, 255, 255), -1)

angle = 0
position = 0

old_time = time.time()

mp_face_mesh = mp.solutions.face_mesh
face_img = None

with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_img = image[int(face_landmarks.landmark[54].y * image.shape[0]) : int(face_landmarks.landmark[365].y * image.shape[0]),
                          int(face_landmarks.landmark[54].x * image.shape[1]) : int(face_landmarks.landmark[288].x * image.shape[1])]


            
            rotated_image = np.copy(image_template)

            if not(face_img is None):
                
                position = int(math.sin(time.time() * motion_speed) * amplitude + k)

                part = rotated_image[center[1] + position : center[1] + enot.shape[0] + position,
                              center[0] - enot.shape[1] // 2 : center[0] + enot.shape[1] // 2]

                mask_inv = cv2.bitwise_not(enot[:part.shape[0],:,-1])
                temp1 = cv2.bitwise_and(part, part, mask = mask_inv)
                temp2 = cv2.bitwise_and(enot[:part.shape[0],:,:-1], enot[:part.shape[0],:,:-1], mask = enot[:part.shape[0],:,-1])
                try:
                    temp2[face[0][1] : face[1][1], face[0][0] : face[1][0]] = cv2.resize(face_img, face_size)
                except:
                    continue

                rotated_image[center[1] + position : center[1] + enot.shape[0] + position,
                              center[0] - enot.shape[1] // 2 : center[0] + enot.shape[1] // 2] = cv2.add(temp1,temp2) 
                
                
                # Rotate the image
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
                rotated_image = cv2.warpAffine(rotated_image, rotation_matrix, (w, h))

                rotated_image = np.where(image_template==0, 0, rotated_image)
                rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
        
                angle += (time.time() - old_time) * angle_speed
                old_time = time.time()
    
            cam.send(rotated_image)
            cam.sleep_until_next_frame()

            if DEBUG:
                cv2.imshow('Result from the Brugge', rotated_image)
                
                if cv2.waitKey(5) == ord("q"):
                     break

        cap.release()
        cv2.destroyAllWindows()
