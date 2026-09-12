# -*- coding: utf-8 -*-
"""
تشخیص زنده‌ی سن و جنسیت با شبکه‌ی عصبی عمیق از پیش‌آموزش‌دیده
Real-time Age & Gender Detection using pretrained deep neural networks

نصب پیش‌نیازها:
    pip install opencv-python numpy

قبل از اجرا - دانلود مدل‌های از پیش‌آموزش‌دیده:
    این پروژه از مدل‌های معروف و رایگان OpenCV/LearnOpenCV استفاده می‌کنه.
    این ۶ فایل رو دانلود کن و کنار همین فایل پایتون (توی همون پوشه) قرار بده
    (همه از این ریپازیتوری معروف قابل دانلودن:
     https://github.com/spmallick/learnopencv/tree/master/AgeGender):

        opencv_face_detector.pbtxt
        opencv_face_detector_uint8.pb
        age_deploy.prototxt
        age_net.caffemodel
        gender_deploy.prototxt
        gender_net.caffemodel

اجرا:
    python age_gender_detector.py

راهنما:
    - جلوی دوربین بشین، صورتت به‌صورت زنده تشخیص داده می‌شه و روی هر صورت
      یه کادر + بازه‌ی سنی تقریبی + جنسیت نشون داده می‌شه
    - q : خروج

نکته: این مدل‌ها بازه‌ی سنی رو تخمین می‌زنن (نه عدد دقیق)، چون خروجی مدل روی
دسته‌بندی‌های از پیش تعریف‌شده (مثل 8-12, 25-32, ...) آموزش دیده، نه یه عدد پیوسته.
دقتش هم صد‌درصد نیست - این یه دموی آموزشی از یادگیری عمیقه، نه ابزار علمی دقیق.
"""

import cv2
import numpy as np
import os

MODEL_DIR = ""

FACE_PROTO = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
               "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LIST = ["Male", "Female"]

FACE_CONFIDENCE_THRESHOLD = 0.7


def check_models_exist():
    required = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
    missing = [f for f in required if not os.path.isfile(f)]
    if missing:
        print("❌ فایل‌های مدل زیر پیدا نشدن:")
        for f in missing:
            print(f"   - {f}")
        print("\nلطفاً طبق راهنمای بالای فایل، مدل‌ها رو توی پوشه‌ی 'models' دانلود و قرار بده.")
        return False
    return True


def detect_faces(net, frame, conf_threshold=FACE_CONFIDENCE_THRESHOLD):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
    return boxes


def main():
    if not check_models_exist():
        return

    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # روی ویندوز DSHOW معمولاً پایدارتر از backend پیش‌فرضه
    if not cap.isOpened():
        print("❌ دوربین باز نشد. مطمئن شو دوربین آزاده (بسته بودن برنامه‌های دیگه مثل Teams/Zoom) "
              "و دسترسی دوربین برای برنامه‌های دسکتاپ در ویندوز فعاله.")
        return
    print("در حال اجرا... q برای خروج")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        boxes = detect_faces(face_net, frame)

        for (x1, y1, x2, y2) in boxes:
            face = frame[max(0, y1 - 10):min(frame.shape[0], y2 + 10),
                          max(0, x1 - 10):min(frame.shape[1], x2 + 10)]
            if face.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_BUCKETS[age_preds[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + len(label) * 11, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(frame, "q: quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Age & Gender Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
