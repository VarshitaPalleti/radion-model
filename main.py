from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import joblib
import cv2
import numpy as np
import io
from PIL import Image
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input

app = FastAPI(title="Radion Diagnostic API")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
MODEL_PATH = 'lung_cancer_svm_model.pkl'
IMG_SIZE = 240
CLASSES = ['Normal', 'Malignant', 'Benign']

# --- LOAD MODELS ---
print("Loading AI Models into memory...")
svm_model = joblib.load(MODEL_PATH)
feature_extractor = EfficientNetB1(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
print("✅ Models Loaded Successfully!")

# --- HELPER FUNCTIONS ---


def apply_clahe(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- API ENDPOINT ---


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image format.")

    try:
        # Read the uploaded image file
        content = await file.read()
        image_pil = Image.open(io.BytesIO(content)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Preprocess exactly like training
        img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        img_clahe = apply_clahe(img_resized)
        img_pre = preprocess_input(np.expand_dims(img_clahe, axis=0))

        # Extract features and predict
        features = feature_extractor.predict(img_pre, verbose=0)
        pred_idx = svm_model.predict(features)[0]
        conf = svm_model.predict_proba(features)[0][pred_idx] * 100
        pred_name = CLASSES[pred_idx]

        # Return clean JSON response
        return {
            "filename": file.filename,
            "prediction": pred_name.upper(),
            "confidence": f"{conf:.2f}%",
            "is_cancer": pred_name.upper() == "MALIGNANT"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
