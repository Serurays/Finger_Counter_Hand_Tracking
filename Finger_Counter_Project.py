import cv2
import os
import time
from Hand_Tracking_Module import HandDetectorMP

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

folder_path = "Finger_Images"
my_list = os.listdir(folder_path)

overlay_list = []

p_time = 0

detector = HandDetectorMP(detection_con=0.75)

for img_path in my_list:
    image = cv2.imread(f'{folder_path}/{img_path}')
    image = cv2.resize(image, (0, 0), fx=0.75, fy=0.75)
    overlay_list.append(image)

tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        fingers = []

        # Thumb
        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[1]][1]: # right hand
            if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        elif lm_list[tip_ids[0]][1] < lm_list[tip_ids[1]][1]:
            if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 Fingers
        for t_id in range(1, 5):
            if lm_list[tip_ids[t_id]][2] < lm_list[tip_ids[t_id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        total_fingers = fingers.count(1)
        print(total_fingers)

        h, w, c = overlay_list[total_fingers - 1].shape
        img[0:h, 0:w] = overlay_list[total_fingers - 1]

        cv2.rectangle(img, (20, 225), (170, 425), (230, 180, 244), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (0, 0, 0), 25)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (200, 100, 50), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
