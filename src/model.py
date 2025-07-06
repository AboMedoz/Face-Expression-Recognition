import json
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from image_preprocessing import preprocess_img

BASR_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASR_DIR)
DATA = os.path.join(ROOT, 'data', 'dataset', 'images', 'train')
MODEL_PATH = os.path.join(ROOT, 'models')

imgs = []
labels = []

for cat in os.listdir(DATA):
    category_folder = os.path.join(DATA, cat)
    for img_paths_raw in os.listdir(category_folder):
        img_paths = os.path.join(category_folder, img_paths_raw)
        img = cv2.imread(img_paths)
        img = preprocess_img(img, -1)
        imgs.append(img)
        labels.append(cat)
    print(f"Finished Folder {cat}")  # Debug

# Sanity check
print(len(imgs))
print(len(labels))

class_names = sorted(set(labels))
class_names_to_index = {name: idx for idx, name in enumerate(class_names)}
int_labels = [class_names_to_index[label] for label in labels]

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(int_labels),
    y=int_labels
)
class_weights = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights}")

x = np.array(imgs)
y = to_categorical(np.array(int_labels), num_classes=7)
x, y = shuffle(x, y, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
datagen = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
)
datagen.fit(x_train)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)
model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=50, validation_data=(x_test, y_test),
          callbacks=[early_stop, reduce_lr], class_weight=class_weights)

_, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}")

model.save(os.path.join(MODEL_PATH, 'face_expression_recognition_model.keras'))
with open(os.path.join(MODEL_PATH, 'class_names.json'), 'w') as f:
    json.dump(class_names, f)
