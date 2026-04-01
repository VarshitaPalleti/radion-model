import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- CONFIGURATION ---
DATA_DIR = "data" 
IMG_SIZE = 240
CLASSES = ['normal', 'malignant', 'benign'] 

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def apply_clahe(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except Exception as e:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("\n--- 1. LOADING & AUGMENTING DATA ---")

def load_and_augment_data(data_dir, classes):
    images = []
    labels = []
    
    for label in classes:
        path = os.path.join(data_dir, label)
        if not os.path.exists(path):
            print(f"⚠️ Warning: Folder '{path}' not found.")
            continue
            
        class_num = classes.index(label)
        print(f"Loading and augmenting images from: {label}...")
        
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path)
                
                if img_array is None: continue

                # Standard Preprocessing
                img_enhanced = apply_clahe(img_array)
                new_array = cv2.resize(img_enhanced, (IMG_SIZE, IMG_SIZE))
                
                # 1. Add Original Image
                images.append(new_array)
                labels.append(class_num)
                
                # 2. Add Augmented Image (Horizontal Flip)
                # Lungs are roughly symmetric. Flipping them doubles our dataset 
                # and forces the model to generalize better to new hospital scans.
                img_flipped = cv2.flip(new_array, 1)
                images.append(img_flipped)
                labels.append(class_num)
                
            except Exception as e:
                pass

    return np.array(images), np.array(labels)

X, y = load_and_augment_data(DATA_DIR, CLASSES)

if len(X) == 0:
    print("❌ Error: No images loaded. Check your folder structure!")
    exit()

print(f"✅ Total Images Loaded (After Augmentation): {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- 2. FEATURE EXTRACTION (EfficientNetB1) ---")
feature_extractor = EfficientNetB1(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)

X_train_pre = preprocess_input(X_train)
X_test_pre = preprocess_input(X_test)

print("Extracting features... (This might take a minute)")
X_train_features = feature_extractor.predict(X_train_pre, verbose=1)
X_test_features = feature_extractor.predict(X_test_pre, verbose=1)

print("\n--- 3. TRAINING SVM (WITH CLASS BALANCING) ---")
# 'class_weight=balanced' forces the SVM to penalize mistakes on minority classes (like Benign)
# much more heavily, preventing it from just guessing "Malignant" all the time.
svm_model = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced') 
svm_model.fit(X_train_features, y_train)

print("\n--- 4. RESULTS ---")
prediction = svm_model.predict(X_test_features)
acc = accuracy_score(y_test, prediction)
print(f"🎯 Model Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, prediction, target_names=CLASSES))

joblib.dump(svm_model, 'lung_cancer_svm_model.pkl')
print("💾 SVM Model saved as 'lung_cancer_svm_model.pkl'")