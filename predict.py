import joblib
import cv2
import numpy as np
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = 'lung_cancer_svm_model.pkl'
IMG_SIZE = 240
CLASSES = ['Normal', 'Malignant', 'Benign']

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


print("\n--- LOADING RADION SYSTEM ---\n")
svm_model = joblib.load(MODEL_PATH)
feature_extractor = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(
    IMG_SIZE, IMG_SIZE, 3), pooling='avg')


def test_image(image_path):
    print(f"\nAnalyzing: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not read image."

    img_enhanced = apply_clahe(img)
    img_resized = cv2.resize(img_enhanced, (IMG_SIZE, IMG_SIZE))
    img_pre = preprocess_input(np.expand_dims(img_resized, axis=0))

    features = feature_extractor.predict(img_pre, verbose=0)
    pred_idx = svm_model.predict(features)[0]
    conf = svm_model.predict_proba(features)[0][pred_idx] * 100

    print(f"Result: {CLASSES[pred_idx].upper()} (Confidence: {conf:.2f}%)")


# Run test
test_image("test_image.png")
