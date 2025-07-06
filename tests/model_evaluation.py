import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from Face_Expression_Recognition.src.image_preprocessing import preprocess_img

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
VAL_DATA_PATH = os.path.join(ROOT, 'data', 'dataset', 'images', 'validation')
MODEL_DIR_PATH = os.path.join(ROOT, 'models', 'face_expression_recognition_model.keras')

imgs = []
labels = []

for category in os.listdir(VAL_DATA_PATH):
    category_folder = os.path.join(VAL_DATA_PATH, category)
    for img_folders in os.listdir(category_folder):
        img_path = os.path.join(category_folder, img_folders)
        img = cv2.imread(img_path)
        img = preprocess_img(img, -1)
        imgs.append(img)
        labels.append(category)
    print(f"Finished Folder {category_folder}")  # Debug

# sanity checks
print(len(imgs))
print(len(labels))

class_names = sorted(set(labels))
class_names_to_index = {name: idx for idx, name in enumerate(class_names)}
int_labels = [class_names_to_index[label] for label in labels]

x = np.array(imgs)
y = to_categorical(int_labels, num_classes=7)

model = load_model(MODEL_DIR_PATH)
_, accuracy = model.evaluate(x, y)
print(f"Accuracy: {accuracy * 100:.2f}")

