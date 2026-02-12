import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. SETUP ---
MODEL_PATH = 'lung_cancer_svm_model.pkl'
IMG_SIZE = 240
CLASSES = ['Normal', 'Malignant', 'Benign'] 

print("--- Loading System ---")
# Load SVM
svm_model = joblib.load(MODEL_PATH)

# Load EfficientNet
feature_extractor = EfficientNetB1(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)

# Setup CLAHE (Must match training!)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def apply_clahe(image):
    # Convert RGB to LAB, apply CLAHE, convert back
    # LIME sends images as RGB (0-255)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# --- 2. THE CRITICAL WRAPPER FUNCTION ---
# LIME will send a batch of perturbed images here.
# We must process them all and return probabilities.
def predict_fn(images):
    # images is a numpy array of shape (Batch_Size, 240, 240, 3)
    
    # A. Preprocess all images in the batch
    # Note: LIME sends float64 images, we need uint8 for OpenCV/CLAHE
    processed_images = []
    for img in images:
        # Convert back to uint8 for CLAHE
        img_uint8 = img.astype(np.uint8) 
        img_clahe = apply_clahe(img_uint8)
        processed_images.append(img_clahe)
    
    processed_images = np.array(processed_images)
    
    # B. EfficientNet Preprocessing
    # EfficientNet expects specific scaling (-1 to 1 or 0 to 255 depending on version)
    # We use the built-in preprocess_input
    images_pre = preprocess_input(processed_images)
    
    # C. Extract Features (The Eye)
    features = feature_extractor.predict(images_pre, verbose=0)
    
    # D. Get Probabilities (The Brain)
    # Returns array like [[0.1, 0.9, 0.0], [0.8, 0.2, 0.0]...]
    probs = svm_model.predict_proba(features)
    
    return probs

# --- 3. RUN LIME ---
def explain_image(image_path):
    print(f"Analyzing {image_path} with LIME...")
    
    # Read Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return
        
    # Resize to model size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Convert BGR to RGB (LIME expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize LIME Explainer
    explainer = lime_image.LimeImageExplainer()

    # Run Explanation
    # hide_color=0 means "turn superpixels black" when hiding them
    # num_samples=1000 means "try 1000 variations" (Higher = More accurate but slower)
    explanation = explainer.explain_instance(
        img_rgb, 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000 
    )

    # Get the image and mask for the top prediction
    # positive_only=True means "Show me what SUPPORTS the cancer decision"
    # num_features=5 means "Show me the top 5 most important superpixels"
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )

    # Visualize
    # mark_boundaries draws yellow lines around the important regions
    img_boundry = mark_boundaries(temp / 255.0, mask)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_boundry)
    plt.title(f"LIME Explanation for: {explanation.top_labels[0]}")
    plt.axis('off')
    plt.show()
    
    # Optional: Save it
    plt.imsave("lime_result.png", img_boundry)
    print("✅ Explanation saved as 'lime_result.png'")

# --- TEST IT ---
# Make sure you have a test image ready
explain_image("test_image.jpg")