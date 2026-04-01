from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB1
import tensorflow as tf
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import joblib
import cv2
import numpy as np
import io
import base64
from PIL import Image
from datetime import datetime
import uuid
import matplotlib
matplotlib.use('Agg')  # CRITICAL for server

app = FastAPI(title="Radion Diagnostic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = 'lung_cancer_svm_model.pkl'
IMG_SIZE = 240
CLASSES = ['Normal', 'Malignant', 'Benign']

print("Loading AI Models into memory...")
svm_model = joblib.load(MODEL_PATH)
feature_extractor = EfficientNetB1(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)
target_layer = feature_extractor.get_layer('top_activation')
activation_model = tf.keras.models.Model(
    inputs=feature_extractor.input, outputs=target_layer.output)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
print("✅ Models Loaded!")


def apply_clahe(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def generate_clinical_report_base64(img_bgr, pred_name, confidence):
    # Prepare Images
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_clahe = apply_clahe(img_resized)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
    img_pre = preprocess_input(np.expand_dims(img_clahe, axis=0))

    # Score-CAM Logic
    activations = activation_model.predict(img_pre, verbose=0)[0]
    num_channels = activations.shape[-1]
    target_idx = CLASSES.index(pred_name)

    weights = []
    batch_size = 64
    for i in range(0, num_channels, batch_size):
        batch_masked = []
        channels_in_batch = min(batch_size, num_channels - i)
        for j in range(channels_in_batch):
            ch_idx = i + j
            act_map = cv2.resize(
                activations[:, :, ch_idx], (IMG_SIZE, IMG_SIZE))
            act_map_norm = act_map - act_map.min()
            if act_map_norm.max() > 0:
                act_map_norm /= act_map_norm.max()
            batch_masked.append(
                img_clahe * np.expand_dims(act_map_norm, axis=-1))

        batch_features = feature_extractor.predict(
            preprocess_input(np.array(batch_masked)), verbose=0)
        batch_probs = svm_model.predict_proba(batch_features)
        for prob in batch_probs:
            weights.append(prob[target_idx])

    cam = np.zeros(
        (activations.shape[0], activations.shape[1]), dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, cam_heatmap, 0.4, 0)

    # Generate Matplotlib Plot in RAM (Fixed clinical style and portrait height)
    fig = plt.figure(figsize=(8, 12))
    fig.patch.set_facecolor('#ffffff')

    scan_id = str(uuid.uuid4()).split('-')[0].upper()
    plt.figtext(0.5, 0.96, "RADION MODEL REPORT",
                ha="center", fontsize=20, fontweight='heavy', color='#1a365d')
    plt.figtext(0.5, 0.92, f"Scan ID: {scan_id}  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ha="center", fontsize=11, color='#4a5568')
    plt.figtext(0.5, 0.89, "-"*80, ha="center", color='#cbd5e0')

    ax = fig.add_axes([0.15, 0.45, 0.7, 0.38])
    ax.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    rect = plt.Rectangle((0, 0), 1, 1, fill=False,
                         color="#2b6cb0", linewidth=3, transform=ax.transAxes)
    ax.add_patch(rect)
    plt.figtext(0.5, 0.42, "Diagnostic Activation Map (Score-CAM)",
                ha="center", fontsize=12, fontweight='bold', color='#2d3748')

    text_color = '#c53030' if pred_name.lower() == 'malignant' else '#2f855a'
    bg_color = '#fff5f5' if pred_name.lower() == 'malignant' else '#f0fff4'
    report_text = (
        f"CLINICAL FINDINGS\n{'='*30}\n\nPRIMARY PREDICTION : {pred_name.upper()}\nCONFIDENCE SCORE   : {confidence:.2f}%\n\nAlgorithm Used     : Radion Model\n")
    plt.figtext(0.5, 0.25, report_text, ha="center", va="center", fontsize=13, bbox={
                "boxstyle": "round,pad=1.5", "facecolor": bg_color, "edgecolor": text_color, "linewidth": 2}, color=text_color, fontweight='bold', family='monospace')
    plt.figtext(0.5, 0.08, "This report is generated by an AI assistant and is for informational purposes only.",
                ha="center", fontsize=9, color='#718096')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        content = await file.read()
        image_pil = Image.open(io.BytesIO(content)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        img_clahe = apply_clahe(img_resized)
        img_pre = preprocess_input(np.expand_dims(img_clahe, axis=0))

        features = feature_extractor.predict(img_pre, verbose=0)
        pred_idx = svm_model.predict(features)[0]
        conf = svm_model.predict_proba(features)[0][pred_idx] * 100
        pred_name = CLASSES[pred_idx]

        report_base64 = generate_clinical_report_base64(
            img_bgr, pred_name, conf)

        return {
            "prediction": pred_name.upper(),
            "confidence": f"{conf:.2f}%",
            "is_cancer": pred_name == "Malignant",
            "report_image_base64": report_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
