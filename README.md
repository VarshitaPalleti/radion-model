# Radion Model for 🫁 Lung Cancer Detection

**Radion** is a clinical-grade, hybrid artificial intelligence microservice designed to detect lung cancer from CT and X-Ray scans. Moving beyond basic classification, this system prioritizes **Explainable AI (XAI)** to combat medical "shortcut learning," generating professional, interpretable clinical reports that prove _why_ the AI made its decision.

---

## 🏗️ Architecture & Technology Stack

We specifically engineered a **Hybrid CNN-SVM Architecture** rather than a standard end-to-end Deep Learning model to maximize accuracy on constrained medical datasets while maintaining deployment efficiency.

| Component               | Technology                                | Why We Chose It                                                                                                                                                               |
| :---------------------- | :---------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Feature Extractor**   | **EfficientNetB1** (TensorFlow/Keras)     | Uses compound scaling to extract complex spatial features (edges, textures) with a very small memory footprint. Ideal for serverless cloud deployment.                        |
| **Classifier**          | **Support Vector Machine** (Scikit-Learn) | Replaces the standard Softmax layer. SVMs with an RBF kernel mathematically separate high-dimensional data (1280 features) more cleanly on smaller medical datasets.          |
| **Image Preprocessing** | **OpenCV (CLAHE)**                        | Raw medical scans often have washed-out contrast. CLAHE (Contrast Limited Adaptive Histogram Equalization) boosts local nodule visibility without exploding background noise. |
| **API Framework**       | **FastAPI**                               | Lightning-fast, natively asynchronous, handles effortlessly, and auto-generates testing documentation.                                                                        |

---

## 🔬 Core Methodologies & The XAI Journey

Achieving 99.8% accuracy on a validation set is easy; proving the model isn't "cheating" is hard. During development, we encountered and solved severe **Shortcut Learning** (Dataset Bias).

1.  **The "Rib/Heart" Bias:** Initial XAI tests (using LIME) revealed the model was achieving high accuracy by looking at the patient's heart size and chest wall density rather than the lung parenchyma.
2.  **Overcoming Background Noise:** Standard geometric masking and thresholding failed due to the identical pixel intensities of room air and lung air.

---

## 📂 Repository Structure

```text
radion-model/
│
├── data/ # (Ignored in Git) Raw Kaggle Dataset
├── lung_cancer_svm_model.pkl # The trained SVM Brain
│
├── train_model.py # Script: Preprocesses data (CLAHE) & trains hybrid model
├── predict.py # Script: Local CLI testing tool for single images
├── main.py # Script: The FastAPI server (Core App)
│
├── requirements.txt # Python dependencies
└── README.md

```

---

## 🚀 Local Installation & Execution

### 1. Prerequisites

Ensure you have Python 3.9+ installed.

### 2. Setup Environment

Clone the repository and install the heavy machine learning dependencies.
_(Note: We use `opencv-python-headless` to prevent GUI thread crashes on servers)._

```bash
git clone https://github.com/VarshitaPalleti/radion-model.git
cd radion-model
pip install -r requirements.txt
```

### 3. Run the Microservice

Boot up the FastAPI server locally.

```bash
python main.py
```

_The server will take ~5 seconds to load the TensorFlow and SVM models into RAM. It will run on `http://localhost:8000`._

---

## 📡 API Documentation

FastAPI automatically generates an interactive testing interface.
Navigate to: **`http://localhost:8000/docs`**

### Endpoint: `POST /predict`

Accepts a raw image file, processes it through the pipeline, runs the Score-CAM XAI algorithm, and generates a clinical report encoded in Base64.

**Request:**

- `Content-Type`: `multipart/form-data`
- `Body`: `file` (Image file: .jpg, .png)

**Response (JSON):**

```json
{
  "filename": "test_image.jpg",
  "prediction": "MALIGNANT",
  "confidence": "99.83%",
  "is_cancer": true
}
```

---

_Disclaimer: Radion is an educational/research tool. It is not FDA-approved and does not replace professional radiologic diagnosis._
