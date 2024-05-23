import cv2
import numpy as np
import math
import time

angle_speed = 100
motion_speed = 10
position_limits = [-100, -50]
face = ((180, 50), (325, 200))

amplitude = abs(position_limits[1] - position_limits[0]) / 2
k = (position_limits[0] + position_limits[1]) / 2

enot = cv2.imread('../images/enot.png', -1)
enot = cv2.resize(enot, (450, 450))

image = np.zeros((500, 500, 3), dtype=np.uint8)

h, w = image.shape[:2]
center = (w // 2, h // 2)

side_length = 50
radius = 200

cv2.circle(image, center, radius, (255, 255, 255), -1)

angle = 0
position = 0

old_time = time.time()
while True:
    rotated_image = np.copy(image)
    
    position = int(math.sin(time.time() * motion_speed) * amplitude + k)

    part = rotated_image[center[1] + position : center[1] + enot.shape[0] + position,
                  center[0] - enot.shape[1] // 2 : center[0] + enot.shape[1] // 2]

    mask_inv = cv2.bitwise_not(enot[:part.shape[0],:,-1])
    temp1 = cv2.bitwise_and(part, part, mask = mask_inv)
    temp2 = cv2.bitwise_and(enot[:part.shape[0],:,:-1], enot[:part.shape[0],:,:-1], mask = enot[:part.shape[0],:,-1])
    temp2[face[0][1] : face[1][1], face[0][0] : face[1][0]] = [0, 0, 255]

    rotated_image[center[1] + position : center[1] + enot.shape[0] + position,
                  center[0] - enot.shape[1] // 2 : center[0] + enot.shape[1] // 2] = cv2.add(temp1,temp2) 
    
    
    # Rotate the image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(rotated_image, rotation_matrix, (w, h))

    rotated_image = np.where(image==0, 0, rotated_image)
    
    cv2.imshow('Result from the Brugge', rotated_image)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    angle += (time.time() - old_time) * angle_speed
    old_time = time.time()

cv2.destroyAllWindows()
