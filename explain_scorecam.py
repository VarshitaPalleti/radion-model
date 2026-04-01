import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from datetime import datetime
import uuid

# --- 1. CONFIGURATION & SETUP ---
MODEL_PATH = 'lung_cancer_svm_model.pkl'
IMG_SIZE = 240
CLASSES = ['Normal', 'Malignant', 'Benign']

print("--- Loading System for Score-CAM ---")
svm_model = joblib.load(MODEL_PATH)

feature_extractor = EfficientNetB1(
    weights='imagenet', include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg'
)

target_layer = feature_extractor.get_layer('top_activation')
activation_model = tf.keras.models.Model(
    inputs=feature_extractor.input, outputs=target_layer.output)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except Exception as e:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- 2. SCORE-CAM ALGORITHM ---


def generate_scorecam(image_path):
    print(f"\nAnalyzing {image_path} with Score-CAM...")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return None

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_clahe = apply_clahe(img_resized)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)

    img_pre = preprocess_input(np.expand_dims(img_clahe, axis=0))

    features = feature_extractor.predict(img_pre, verbose=0)
    probs = svm_model.predict_proba(features)[0]
    target_class_idx = np.argmax(probs)
    target_class_name = CLASSES[target_class_idx]

    activations = activation_model.predict(img_pre, verbose=0)[0]
    num_channels = activations.shape[-1]

    weights = []
    batch_size = 64

    for i in range(0, num_channels, batch_size):
        batch_masked_imgs = []
        channels_in_batch = min(batch_size, num_channels - i)

        for j in range(channels_in_batch):
            ch_idx = i + j
            act_map = activations[:, :, ch_idx]
            act_map_resized = cv2.resize(act_map, (IMG_SIZE, IMG_SIZE))
            act_map_norm = act_map_resized - act_map_resized.min()
            map_max = act_map_norm.max()
            if map_max > 0:
                act_map_norm = act_map_norm / map_max

            masked_img = img_clahe * np.expand_dims(act_map_norm, axis=-1)
            batch_masked_imgs.append(masked_img)

        batch_masked_imgs = np.array(batch_masked_imgs)
        batch_pre = preprocess_input(batch_masked_imgs)

        batch_features = feature_extractor.predict(batch_pre, verbose=0)
        batch_probs = svm_model.predict_proba(batch_features)

        for prob in batch_probs:
            weights.append(prob[target_class_idx])

    cam = np.zeros(
        (activations.shape[0], activations.shape[1]), dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam - cam.min()
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max

    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, cam_heatmap, 0.4, 0)

    return superimposed_img, target_class_name, probs[target_class_idx]

# --- 3. PROFESSIONAL CLINICAL REPORT GENERATOR (Fixed Layout) ---


def generate_medical_report(image_path, output_filename="patient_diagnostic_report.png"):
    result = generate_scorecam(image_path)
    if result is None:
        return

    cam_img, pred, conf = result

    # 1. Setup the Clinical Dashboard Canvas (Increased Height to 12 inches for padding)
    fig = plt.figure(figsize=(8, 12))
    fig.patch.set_facecolor('#ffffff')

    # 2. Header Information (Shifted Downward slightly)
    scan_id = str(uuid.uuid4()).split('-')[0].upper()
    plt.figtext(0.5, 0.96, "RADION MODEL REPORT", ha="center",
                fontsize=20, fontweight='heavy', color='#1a365d', family='sans-serif')
    plt.figtext(0.5, 0.92, f"Scan ID: {scan_id}  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Modality: CT/X-Ray",
                ha="center", fontsize=11, color='#4a5568')
    plt.figtext(0.5, 0.89, "-"*80, ha="center", color='#cbd5e0')

    # 3. Score-CAM Image Plot (Single, large, centered, with ample top and bottom space)
    # [left, bottom, width, height]
    ax = fig.add_axes([0.15, 0.45, 0.7, 0.38])
    ax.imshow(cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    # Draw a neat clinical border around the image
    rect = plt.Rectangle((0, 0), 1, 1, fill=False,
                         color="#2b6cb0", linewidth=3, transform=ax.transAxes)
    ax.add_patch(rect)
    plt.figtext(0.5, 0.42, "Diagnostic Activation Map (Score-CAM)",
                ha="center", fontsize=12, fontweight='bold', color='#2d3748')

    # 4. Diagnostic Text Block (Dynamic Colors, updated wording)
    text_color = '#c53030' if pred.lower() == 'malignant' else '#2f855a'
    bg_color = '#fff5f5' if pred.lower() == 'malignant' else '#f0fff4'

    report_text = (
        f"CLINICAL FINDINGS\n"
        f"{'='*30}\n\n"
        f"PRIMARY PREDICTION : {pred.upper()}\n"
        f"CONFIDENCE SCORE   : {conf*100:.2f}%\n\n"
        f"Algorithm Used     : Radion Model (EfficientNetB1 + SVM)\n"
        f"Highlight Method   : Gradient-Free Activation (Score-CAM)\n"
    )

    plt.figtext(0.5, 0.25, report_text, ha="center", va="center",
                fontsize=13, bbox={"boxstyle": "round,pad=1.5", "facecolor": bg_color, "edgecolor": text_color, "linewidth": 2},
                color=text_color, fontweight='bold', family='monospace')

    # Footer (Ample space from bottom and text box)
    plt.figtext(0.5, 0.08, "This report is generated by an AI assistant and is for informational purposes only.\nIt does not replace a professional medical diagnosis.",
                ha="center", fontsize=9, color='#718096')

    # 5. Adjust layout and Save
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Professional Medical Report saved as: '{output_filename}'")


# --- RUN THE GENERATOR ---
# Point this to your image
test_image_file = "test_image.png"  # Update with your exact filename
generate_medical_report(test_image_file)
