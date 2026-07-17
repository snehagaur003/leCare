import io
import base64
import datetime as dt
import streamlit as slt
import numpy as np
import os 
from google import genai
import pandas as pd
import cv2
from PIL import Image, ImageOps

client = genai.Client(api_key=os.getenv("AIzaSyBI_zztaKWf8R7YvmvtFurgH8sIOtwDh8A"))
def validate_leaf(image: Image.Image):

    prompt = """
                    You are validating images for a plant disease detection system.
                    Rules:
                    1. The image must clearly contain at least one plant leaf.
                    2. The leaf should occupy a significant portion of the image.
                    3. Reject humans, hands, animals, fruits, flowers, vehicles, buildings, food, phones, or any object that is not primarily a plant leaf.
                    4. If you are not confident, answer NO.
                    Respond ONLY with one word:
                    YES or NO
                    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, prompt]
    )

    import json

    try:
        result = json.loads(response.text.strip())
        return result["valid"], result["reason"]
    except:
        return False, "Unable to validate image."

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

leaf_detector = MobileNetV2(weights="imagenet")

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================
slt.set_page_config(
    page_title="leCare - Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "./best_plant_model.keras"
IMG_SIZE = (224, 224)

# ============================================================
# CLASS NAMES & METADATA
# ============================================================
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]
treatments = {
    "Apple___Apple_scab": "Prune infected branches, rake and destroy fallen leaves, and apply a protectant fungicide (e.g. captan or myclobutanil) starting at bud break.",
    "Apple___Black_rot": "Remove mummified fruit and cankers, prune out dead wood, and apply fungicide during the growing season.",
    "Apple___Cedar_apple_rust": "Remove nearby juniper/cedar hosts if possible and apply preventive fungicide sprays from pink bud through petal fall.",
    "Apple___healthy": "No disease detected — continue regular watering, pruning, and monitoring.",
    "Cherry_(including_sour)___Powdery_mildew": "Improve air circulation via pruning, avoid overhead watering, and apply sulfur or potassium bicarbonate fungicide.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops, use resistant hybrids, and apply a foliar fungicide if disease pressure is high.",
    "Corn_(maize)___Common_rust_": "Plant resistant hybrids and apply fungicide if pustules appear early and coverage is high.",
    "Corn_(maize)___Northern_Leaf_Blight": "Rotate with non-host crops, till residue, and apply fungicide at first sign of lesions.",
    "Grape___Black_rot": "Remove mummified berries and infected canes; apply fungicide from early shoot growth through veraison.",
    "Grape___Esca_(Black_Measles)": "Prune out and destroy infected wood; avoid large pruning wounds during wet weather; no curative fungicide exists.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Improve canopy airflow, remove infected leaves, and apply copper-based fungicide.",
    "Orange___Haunglongbing_(Citrus_greening)": "Remove and destroy infected trees, control the Asian citrus psyllid vector, and use certified disease-free planting stock.",
    "Peach___Bacterial_spot": "Plant resistant varieties, avoid overhead irrigation, and apply copper sprays during dormancy.",
    "Pepper,_bell___Bacterial_spot": "Use certified disease-free seed, rotate crops, and apply copper-based bactericide.",
    "Potato___Early_blight": "Remove infected leaves, rotate crops away from nightshades, and apply chlorothalonil or mancozeb fungicide.",
    "Potato___Late_blight": "Destroy infected plants immediately, avoid overhead watering, and apply a protectant fungicide before wet weather.",
    "Squash___Powdery_mildew": "Space plants for airflow, water at the base, and apply sulfur or neem oil at first sign of white patches.",
    "Strawberry___Leaf_scorch": "Remove infected leaves after harvest, avoid overhead watering, and apply fungicide in early spring.",
    "Tomato___Bacterial_spot": "Use disease-free seed/transplants, avoid working plants when wet, and apply copper-based bactericide.",
    "Tomato___Early_blight": "Remove lower infected leaves, mulch to prevent soil splash, and apply fungicide on a 7–10 day schedule.",
    "Tomato___Late_blight": "Remove and destroy infected foliage immediately and apply fungicide before the next rain.",
    "Tomato___Leaf_Mold": "Improve greenhouse/garden ventilation, reduce humidity, and apply fungicide if conditions stay humid.",
    "Tomato___Septoria_leaf_spot": "Remove infected lower leaves, mulch, avoid overhead watering, and apply fungicide.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Rinse leaves with water, introduce predatory mites, or apply insecticidal soap/neem oil.",
    "Tomato___Target_Spot": "Remove infected debris, rotate crops, and apply a labeled fungicide.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies with row covers or insecticide, remove infected plants, and use resistant varieties.",
    "Tomato___Tomato_mosaic_virus": "Remove and destroy infected plants, disinfect tools between plants, and avoid tobacco contact before handling plants.",
}


def get_treatment(predict_class: str) -> str:
    if predict_class in treatments:
        return treatments[predict_class]
    if predict_class.endswith("healthy"):
        return "No disease detected — continue regular care and monitoring."
    name = predict_class.split("___")[-1].replace("_", " ")
    return (
        f"Detected condition: {name}. Remove and destroy visibly infected leaves, "
        "avoid overhead watering, ensure good airflow between plants, and consult "
        "a local agricultural extension office for a fungicide/bactericide labeled "
        "for this specific disease."
    )


# ============================================================
# CACHED RESOURCES
# ============================================================
@slt.cache_resource(show_spinner=False)
def load_my_model():
    if not TF_AVAILABLE:
        return None
    return load_model(MODEL_PATH)


@slt.cache_data(show_spinner=False)
def load_css_text(file_name: str) -> str:
    with open(file_name) as f:
        return f.read()


@slt.cache_data(show_spinner=False)
def encode_background(image_path: str) -> str:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, subsampling=0)
    return base64.b64encode(buf.getvalue()).decode()


def load_css(file_name):
    try:
        slt.markdown(f"<style>{load_css_text(file_name)}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


def set_background(image_path: str):
    try:
        encoded = encode_background(image_path)
        bg_rule = f'url("data:image/jpeg;base64,{encoded}")'
    except FileNotFoundError:

        bg_rule = "linear-gradient(135deg, #0f2913 0%, #1B5E20 45%, #2E7D32 100%)"

    slt.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.35)),
                {bg_rule};
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            image-rendering: -webkit-optimize-contrast;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
def inject_result_bar_css():
    slt.markdown(
        """
        <style>
        .result-bar {
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 12px;
            font-size: 1.05rem;
            font-weight: 500;
            color: #eafbea;
        }
        .bar-prediction { background: rgba(27, 94, 32, 0.55); border: 1px solid rgba(76, 175, 80, 0.5); }
        .bar-info       { background: rgba(21, 60, 94, 0.55);  border: 1px solid rgba(66, 133, 189, 0.5); color:#dff1ff;}
        .bar-treatment  { background: rgba(85, 82, 20, 0.55); border: 1px solid rgba(180, 170, 60, 0.5); color:#fbf7df;}
        .bar-warning    { background: rgba(120, 60, 20, 0.55); border: 1px solid rgba(220, 140, 60, 0.5); color:#ffe9d6;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def result_bar(text: str, kind: str = "info"):
    css_class = {"prediction": "bar-prediction", "info": "bar-info", "treatment": "bar-treatment", "warning": "bar-warning"}[kind]
    slt.markdown(f"<div class='result-bar {css_class}'>{text}</div>", unsafe_allow_html=True)


def severity_badge_html(level: str) -> str:
    badge_class = {"Mild": "badge-mild", "Moderate": "badge-moderate", "Severe": "badge-severe"}.get(level, "badge-mild")
    return f"<span class='severity-badge {badge_class}'>{level}</span>"


# ============================================================
# IMAGE ANALYSIS
# ============================================================
def get_severity(leaf_image_rgb: np.ndarray):
    """Runs on the FULL-RESOLUTION uploaded image (not the 224x224 model
    input) so the percentage and overlay are accurate and crisp, not
    computed from a blurry downscaled thumbnail."""
    hsv = cv2.cvtColor(leaf_image_rgb, cv2.COLOR_RGB2HSV)

    # brown/yellow blight & rust tones
    lower_brown = np.array([5, 50, 50])
    upper_brown = np.array([30, 255, 255])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # dark necrotic / black-spot tissue
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    mask = cv2.bitwise_or(mask_brown, mask_dark)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    disease_pixels = cv2.countNonZero(mask)
    total_pixels = leaf_image_rgb.shape[0] * leaf_image_rgb.shape[1]
    severity_percent = (disease_pixels / total_pixels) * 100

    if severity_percent < 10:
        level = "Mild"
    elif severity_percent < 30:
        level = "Moderate"
    else:
        level = "Severe"

    return severity_percent, level, mask


def make_overlay(leaf_image_rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Highlights the detected diseased regions in red on the full-res image."""
    overlay = leaf_image_rgb.copy()
    overlay[mask > 0] = [255, 60, 60]
    blended = cv2.addWeighted(leaf_image_rgb, 0.6, overlay, 0.4, 0)
    return Image.fromarray(blended)


def predict_image(model, pil_image: Image.Image):
    resized = pil_image.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]

    top3_idx = np.argsort(preds)[-3:][::-1]
    top3 = [(class_names[i], float(preds[i]) * 100) for i in top3_idx]

    return top3


# ============================================================
# SESSION STATE
# ============================================================
if "page" not in slt.session_state:
    slt.session_state.page = "home"
if "history" not in slt.session_state:
    slt.session_state.history = []  # list of dicts

load_css("./css/style.css")
inject_result_bar_css()

PAGE_TO_LABEL = {"home": "Home", "predict": "Detect", "history": "History", "about": "About"}
LABEL_TO_PAGE = {v: k for k, v in PAGE_TO_LABEL.items()}
NAV_OPTIONS = ["Home", "Detect", "History", "About"]

# ============================================================
# SIDEBAR (present on every page)
# ============================================================

with slt.sidebar:
    with slt.sidebar:

        slt.markdown("## 🌿 leCare")

        if slt.button("🏠 Home", use_container_width=True):
            slt.session_state.page = "home"

        if slt.button("🔍 Detect", use_container_width=True):
            slt.session_state.page = "predict"

        if slt.button("📊 History", use_container_width=True):
            slt.session_state.page = "history"

        if slt.button("ℹ️ About", use_container_width=True):
            slt.session_state.page = "about"

    slt.divider()

    confidence_threshold = slt.slider(
        "Confidence alert threshold (%)",
        30,
        90,
        60,
    )

    slt.caption(
        f"Predictions below {confidence_threshold}% will be flagged as low-confidence."
    )

    if slt.session_state.history:
        slt.divider()
        slt.caption(f"📊 {len(slt.session_state.history)} scan(s) this session")

        if slt.button("Clear history", use_container_width=True):
            slt.session_state.history = []
            slt.rerun()

page = slt.session_state.page


def back_to_home_button():
    if slt.button("🏠 Back to Home", use_container_width=False):
        slt.session_state.page = "home"
        slt.rerun()


# ============================================================
# HOME
# ============================================================
if page == "home":
    set_background("./images/obg.jpg")
    left, right = slt.columns([1, 1])

    with right:
        slt.markdown("<h1 class='main-title'>🌿 leCare</h1>", unsafe_allow_html=True)
        slt.markdown("<h3 class='sub-title'>Smart Plant Disease Detection System</h3>", unsafe_allow_html=True)
        slt.markdown(
            "<p class='description'>Upload a photo of a leaf and leCare identifies the disease, "
            "estimates severity from the actual affected area, and gives you a treatment plan — "
            "in seconds.</p>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = slt.columns(3)
        c1.markdown("<div class='stat-pill'>🌱 37 classes</div>", unsafe_allow_html=True)
        c2.markdown("<div class='stat-pill'>⚡ Instant results</div>", unsafe_allow_html=True)
        c3.markdown("<div class='stat-pill'>📋 Treatment tips</div>", unsafe_allow_html=True)

        slt.write("")
        if slt.button("🌿 Let me check your leaf!", use_container_width=True):
            slt.session_state.page = "predict"
            slt.rerun()

# DETECT

elif page == "predict":
    set_background("./images/page2.jpg")
    back_to_home_button()
    slt.title("🔍 Plant Disease Detection")
    slt.write("Upload one or more leaf images to detect diseases, severity, and get treatment advice.")

    if not TF_AVAILABLE:
        slt.error(
            "TensorFlow isn't available in this environment, so predictions are disabled. "
            "Install `tensorflow` to enable the model."
        )
        
    model = load_my_model() if TF_AVAILABLE else None
    if TF_AVAILABLE and model is None:
        slt.error(f"Could not load the model from `{MODEL_PATH}`. Check the file path.")

    upload_files = slt.file_uploader(
        "Choose leaf image(s)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if upload_files:
        run = slt.button(f"Analyze {len(upload_files)} image(s)", use_container_width=True, disabled=model is None)

        if run:
            progress = slt.progress(0, text="Starting analysis…")

            for i, upload_file in enumerate(upload_files):
                try:
                    original_image = Image.open(upload_file)
                    original_image = ImageOps.exif_transpose(original_image).convert("RGB")
                except Exception:
                    slt.error(f"Couldn't read `{upload_file.name}` — is it a valid image file?")
                    continue

                full_res_array = np.array(original_image)

                with slt.spinner(f"Analyzing {upload_file.name}…"):
                    valid, reason = validate_leaf(uploaded_image)
                    if not valid:
                        st.error(f"❌ {reason}")
                        st.stop()
                    top3 = predict_image(model, uploaded_image)
                    predict_class, confidence = top3[0]
                    severity_percent, level, mask = get_severity(full_res_array)
                    overlay_img = make_overlay(full_res_array, mask)

                display_name = predict_class.replace("___", " - ").replace("_", " ")
                treatment = get_treatment(predict_class)

                record = {
                    "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "file": upload_file.name,
                    "disease": display_name,
                    "confidence": round(confidence, 2),
                    "severity_pct": round(severity_percent, 2),
                    "severity_level": level,
                    "treatment": treatment,
                }
                slt.session_state.history.append(record)

                # ---- Layout: image on the LEFT, result bars stacked on the RIGHT ----
                with slt.container():
                    slt.markdown(f"#### 📄 {upload_file.name}")
                    img_col, result_col = slt.columns([1, 1])

                    with img_col:
                        tab1, = slt.tabs(["Uploaded Leaf Image"])
                        with tab1:
                            slt.image(
                                original_image,
                                use_container_width=True,
                                caption="Uploaded Leaf Image"
                            )
                    with result_col:
                        result_bar(f"<b>Prediction:</b> {display_name}", kind="prediction")
                        result_bar(f"<b>Affected:</b> {severity_percent:.2f}%", kind="info")
                        result_bar(f"<b>Severity:</b> {severity_badge_html(level)}", kind="info")

                        if confidence < confidence_threshold:
                            result_bar(f"<b>Confidence:</b> {confidence:.2f}% ⚠️ Low confidence — try a clearer, closer photo of a single leaf.", kind="warning")
                        else:
                            result_bar(f"<b>Confidence:</b> {confidence:.2f}%", kind="info")

                        result_bar(f"<b>Treatment:</b> {treatment}", kind="treatment")

                        report_text = (
                            f"leCare Diagnosis Report\n"
                            f"File: {upload_file.name}\n"
                            f"Date: {record['timestamp']}\n"
                            f"Disease: {display_name}\n"
                            f"Confidence: {confidence:.2f}%\n"
                            f"Severity: {level} ({severity_percent:.2f}% of leaf area)\n"
                            f"Treatment: {treatment}\n"
                        )
                        slt.download_button(
                            "⬇️ Download report (.txt)",
                            data=report_text,
                            file_name=f"lecare_report_{upload_file.name.split('.')[0]}.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key=f"dl_{i}_{upload_file.name}",
                        )
                    slt.divider()
                progress.progress((i + 1) / len(upload_files), text=f"Analyzed {i + 1}/{len(upload_files)}")
            progress.empty()
# HISTORY

elif page == "history":
    set_background("./images/page2.jpg")
    back_to_home_button()
    slt.title("📊 Scan History")

    if not slt.session_state.history:
        slt.info("No scans yet this session. Head to **Detect** to analyze a leaf image.")
    else:
        hist_df = pd.DataFrame(slt.session_state.history)
        slt.dataframe(hist_df, use_container_width=True, hide_index=True)

        m1, m2, m3 = slt.columns(3)
        m1.metric("Total scans", len(hist_df))
        m2.metric("Avg. confidence", f"{hist_df['confidence'].mean():.1f}%")
        m3.metric("Healthy vs. diseased", f"{(hist_df['disease'].str.contains('healthy', case=False)).sum()} / {len(hist_df)}")

        csv = hist_df.to_csv(index=False).encode("utf-8")
        slt.download_button("⬇️ Export full history (.csv)", data=csv, file_name="lecare_history.csv", mime="text/csv")


# ============================================================
# ABOUT
# ============================================================
elif page == "about":
    set_background("./images/page2.jpg")
    back_to_home_button()
    slt.title("ℹ️ About leCare")
    slt.markdown(
        """
        Hi there I am LeCare, you smart plant buddy. When you share a leaf with me, I carefully examine its patterns, colors, and textures using artificial intelligence to identify possible diseases. I don't stop at a diagnosis. I estimate how much of the leaf has been affected, measure my confidence in the prediction, and provide practical guidance to help you protect your plants.
        I'm here to make plant care simpler, faster, and more informed. Whether you're a farmer caring for an entire field, a gardener nurturing a backyard, or a student exploring agriculture through technology, I'm designed to support your decisions with clear and accessible information.
        Every scan is an opportunity to detect problems early, reduce crop losses, and encourage healthier plants. As I continue to grow, I'll become an even smarter companion by answering your questions, explaining diseases in simple language, and helping you understand not just what happened to your plant, but why it happened and how you can respond.
        Together, let's give every leaf the attention it deserves.
                """
    )

else:
    slt.session_state.page = "home"
    slt.rerun()
