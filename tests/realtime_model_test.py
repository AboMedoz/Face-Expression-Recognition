import json
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from Face_Expression_Recognition.src.image_preprocessing import preprocess_img

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT, 'models', 'face_expression_recognition_model.keras')
CLASS_NAMES_PATH = os.path.join(ROOT, 'models', 'class_names.json')

model = load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, h, w) in face:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = preprocess_img(roi_gray, 0, False)

        prediction = model.predict(roi_gray)
        expression = class_names[np.argmax(prediction)]

        cv2.putText(frame, f"{expression}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

