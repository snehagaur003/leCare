# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# MODEL_PATH = "best_plant_model.keras"
# IMAGE_PATH = "./image.png"  

# IMAGE_SIZE = 224

# class_names = [
#     "Apple___Apple_scab",
#     "Apple___Black_rot",
#     "Apple___Cedar_apple_rust",
#     "Apple___healthy",
#     "Blueberry___healthy",
#     "Cherry_(including_sour)___Powdery_mildew",
#     "Cherry_(including_sour)___healthy",
#     "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
#     "Corn_(maize)___Common_rust_",
#     "Corn_(maize)___Northern_Leaf_Blight",
#     "Corn_(maize)___healthy",
#     "Grape___Black_rot",
#     "Grape___Esca_(Black_Measles)",
#     "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
#     "Grape___healthy",
#     "Orange___Haunglongbing_(Citrus_greening)",
#     "Peach___Bacterial_spot",
#     "Peach___healthy",
#     "Pepper,_bell___Bacterial_spot",
#     "Pepper,_bell___healthy",
#     "Potato___Early_blight",
#     "Potato___Late_blight",
#     "Potato___healthy",
#     "Raspberry___healthy",
#     "Soybean___healthy",
#     "Squash___Powdery_mildew",
#     "Strawberry___Leaf_scorch",
#     "Strawberry___healthy",
#     "Tomato___Bacterial_spot",
#     "Tomato___Early_blight",
#     "Tomato___Late_blight",
#     "Tomato___Leaf_Mold",
#     "Tomato___Septoria_leaf_spot",
#     "Tomato___Spider_mites Two-spotted_spider_mite",
#     "Tomato___Target_Spot",
#     "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
#     "Tomato___Tomato_mosaic_virus",
#     "Tomato___healthy"
# ]

# model = load_model(MODEL_PATH)

# img = image.load_img(
#     IMAGE_PATH,
#     target_size=(224, 224)
# )

# img_array = image.img_to_array(img)
# img_array = img_array / 255.0
# img_array = np.expand_dims(img_array, axis=0)

# prediction = model.predict(img_array, verbose=0)

# predicted_index = np.argmax(prediction)

# predicted_class = class_names[predicted_index]

# confidence = np.max(prediction) * 100

# print("Prediction:", predicted_class)
# print(f"Confidence: {confidence:.2f}%")

from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "best_plant_model.keras"
IMAGE_DIR = "./dataset_small/test" 
IMAGE_SIZE = 224

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

model = load_model(MODEL_PATH)

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

for img_path in Path(IMAGE_DIR).iterdir():

    if img_path.suffix.lower() not in image_extensions:
        continue

    try:
        img = image.load_img(
            img_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE)
        )

        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = np.max(prediction) * 100

        print(f"\nImage: {img_path.name}")
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        print(f"Could not process {img_path.name}: {e}")