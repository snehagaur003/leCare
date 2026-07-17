import random
import numpy as np
import tensorflow as tf
import datetime

# ==================================================
# REPRODUCIBILITY
# ==================================================

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# ==================================================
# IMPORTS
# ==================================================

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras import layers, models

# ==================================================
# SETTINGS
# ==================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

TRAIN_DIR = "./dataset_small/train"
VALID_DIR = "./dataset_small/valid"

LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ==================================================
# DATA AUGMENTATION
# ==================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==================================================
# DATA GENERATORS
# ==================================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Number of classes:", train_generator.num_classes)

# ==================================================
# BASE MODEL
# ==================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze everything first
base_model.trainable = True

# Fine tune only last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

print("MobileNetV2 loaded successfully")

# ==================================================
# MODEL
# ==================================================

model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        train_generator.num_classes,
        activation="softmax"
    )
])

print("Model created successfully!")

# ==================================================
# COMPILE
# ==================================================

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(
#         learning_rate=0.0001
#     ),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("Model compiled successfully!")

# ==================================================
# CALLBACKS
# ==================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_plant_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

tensorboard_callback = TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=1,
    write_graph=True,
    update_freq="epoch"
)

# ==================================================
# TRAIN
# ==================================================

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr,
        tensorboard_callback
    ]
)

# ==================================================
# SAVE FINAL MODEL
# ==================================================

model.save("best_plant_model.keras")

print("Training completed!")
print(history.history.keys())