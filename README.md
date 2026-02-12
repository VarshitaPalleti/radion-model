# **Radion Model** for 🫁 Lung Cancer Detection (EfficientNetB1 + SVM)

A hybrid Deep Learning medical imaging project that detects lung cancer from CT scans/X-rays. It combines the feature extraction power of **EfficientNetB1** with the robust classification capability of **Support Vector Machines (SVM)** to achieve high accuracy on small datasets.

## 🚀 Key Features
* **Hybrid Architecture:** Uses a pre-trained CNN (EfficientNet) for vision and SVM for decision making.
* **CLAHE Enhancement:** Implements *Contrast Limited Adaptive Histogram Equalization* to improve visibility of nodules in lung scans.
* **Explainable AI (XAI):** Integrated **LIME** (Local Interpretable Model-agnostic Explanations) to visualize exactly which part of the lung the model is looking at.
* **High Sensitivity:** Optimized to minimize False Negatives (Cancer cases missed).

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Deep Learning:** TensorFlow / Keras (EfficientNetB1)
* **Machine Learning:** Scikit-Learn (SVM)
* **Image Processing:** OpenCV (CLAHE), Pillow
* **Explainability:** LIME

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/VarshitaPalleti/radion-model.git
    cd radion-model
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Data:**
    * **Main Dataset**: Download the [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset) from Kaggle for the _model training_
    * Create a folder named `data` in the root directory.
    * Inside `data`, ensure you have three folders: `Normal`, `Malignant`, `Benign`.
    * **Optional**: Download the [Lung Cancer CT Scans](https://www.kaggle.com/datasets/lalosalamanca1261/lung-cancer-ct-scans) from Kaggle for _model testing_

## 🏃‍♂️ Usage

### 1. Train the Model
Run the training script to process images, apply CLAHE, and train the SVM.
```bash
python train_model.py

```

*Output:* Saves the trained model as `lung_cancer_svm_model.pkl`.

### 2. Predict on a Single Image

Test the model on a specific image file.

```bash
python predict.py

```

*(Note: Update the image path inside `predict.py` before running)*

### 3. Explain the Decision (LIME)

Generate a visual explanation of why the model made its decision.

```bash
python explain_lime.py

```

*Output:* Opens a window showing the superpixels that influenced the prediction (e.g., ribs, nodules, tissue).

## 📊 Results

* **Accuracy:** ~90.5%
* **Malignant Recall:** 100% (No cancer cases missed)
* **Preprocessing:** Images resized to 240x240 and enhanced via CLAHE (ClipLimit=2.0).

## ⚠️ Limitations

* **Shortcut Learning:** LIME analysis reveals the model sometimes relies on anatomical features (like ribs/heart shape) rather than just lung nodules. Future work involves implementing Lung Segmentation (UNet) to isolate the lungs before classification.

## 📝 License

This project is for educational purposes.
