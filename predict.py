import joblib
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = 'lung_cancer_svm_model.pkl'
IMG_SIZE = 240
CLASSES = ['normal', 'malignant', 'benign']

# --- CLAHE SETUP ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final
    except Exception as e:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


print("\n--- 1. LOADING THE SYSTEM ---\n")

print("Loading SVM model...")
svm_model = joblib.load(MODEL_PATH)

print("Loading EfficientNet...\n")
feature_extractor = EfficientNetB1(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)


def predict_image(image_path):
    print(f"\nAnalyzing image: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not read image file."

    # --- MUST MATCH TRAINING PIPELINE ---
    # 1. Apply CLAHE first
    img_enhanced = apply_clahe(img)
    # 2. Resize
    img_resized = cv2.resize(img_enhanced, (IMG_SIZE, IMG_SIZE))

    # 3. Prepare for EfficientNet
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_array)

    # 4. Extract & Predict
    features = feature_extractor.predict(img_preprocessed, verbose=0)

    prediction_index = svm_model.predict(features)[0]
    confidence_scores = svm_model.predict_proba(features)[0]

    result = CLASSES[prediction_index]
    confidence = confidence_scores[prediction_index] * 100

    return f"Result: {result.upper()} (Confidence: {confidence:.2f}%)"


# Test the image
image_to_test = "test_image.jpg"  

try:
    print(predict_image(image_to_test))
except Exception as e:
    print(f"Error: {e}")
    print("Make sure you put a file named 'test_image.jpg' in this folder!")
