from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import joblib
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. SETUP & CONFIG ---
app = FastAPI(title="Lung Cancer Detection API", version="1.0")

# Enable CORS so Express/Next.js can talk to this API safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = 'lung_cancer_svm_model.pkl'
IMG_SIZE = 240
CLASSES = ['Normal', 'Malignant', 'Benign']

# --- 2. LOAD MODELS ON STARTUP ---
print("Loading AI Models... This takes a few seconds.")
svm_model = joblib.load(MODEL_PATH)
feature_extractor = EfficientNetB1(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
print("✅ Models Loaded Successfully!")

# --- 3. HELPER FUNCTIONS ---
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

def process_and_predict(image_bytes):
    # Convert uploaded bytes to an OpenCV image
    image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # 1. Apply CLAHE (Crucial for matching training data)
    img_enhanced = apply_clahe(img_bgr)
    
    # 2. Resize
    img_resized = cv2.resize(img_enhanced, (IMG_SIZE, IMG_SIZE))

    # 3. Prepare for EfficientNet
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_array)

    # 4. Extract Features & Predict
    features = feature_extractor.predict(img_preprocessed, verbose=0)
    prediction_index = svm_model.predict(features)[0]
    confidence_scores = svm_model.predict_proba(features)[0]

    result = CLASSES[prediction_index]
    confidence = confidence_scores[prediction_index] * 100

    return result.upper(), confidence

# --- 4. API ENDPOINTS ---
@app.get("/")
async def root():
    return {"message": "Lung Cancer AI Microservice is running."}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Validate that the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image format (jpg, png, etc.)")

    try:
        # Read the file from the request
        content = await file.read()
        
        # Run prediction
        result, confidence = process_and_predict(content)

        # Return clean JSON to your Express backend
        return {
            "filename": file.filename,
            "prediction": result,
            "confidence": f"{confidence:.2f}%",
            "is_cancer": result == "MALIGNANT"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# --- 5. RUN SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)