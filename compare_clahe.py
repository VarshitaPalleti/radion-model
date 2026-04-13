import cv2
import matplotlib.pyplot as plt

# --- 1. EXACT CLAHE SETUP FROM YOUR PROJECT ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(image):
    """
    Applies CLAHE exactly as it is done in your train_model.py
    Returns an RGB image.
    """
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except Exception as e:
        print(f"CLAHE Error: {e}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- 2. COMPARISON GENERATOR ---


def generate_comparison(image_path):
    print(f"Processing: {image_path}")

    # Read the original image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Error: Could not load '{image_path}'. Check the filename.")
        return

    # Convert original to RGB for Matplotlib
    img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Apply CLAHE (Our function already returns RGB)
    img_rgb_enhanced = apply_clahe(img_bgr)

    # --- SAVE 1: Side-by-Side Presentation Image ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('#ffffff')

    axes[0].imshow(img_rgb_original)
    axes[0].set_title("Before: Original Scan", fontsize=14,
                      fontweight='bold', pad=15)
    axes[0].axis('off')

    axes[1].imshow(img_rgb_enhanced)
    axes[1].set_title("After: CLAHE Enhanced", fontsize=14,
                      fontweight='bold', pad=15)
    axes[1].axis('off')

    plt.tight_layout()
    comparison_filename = "clahe_comparison_report.jpg"
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Side-by-side comparison saved as: '{comparison_filename}'")

    # --- SAVE 2: Raw Enhanced Image ---
    # Convert RGB back to BGR so OpenCV can save it properly
    img_bgr_enhanced = cv2.cvtColor(img_rgb_enhanced, cv2.COLOR_RGB2BGR)
    raw_filename = "clahe_raw_enhanced.jpg"
    cv2.imwrite(raw_filename, img_bgr_enhanced)
    print(f"✅ Raw enhanced image saved as: '{raw_filename}'")


# --- 3. RUN THE SCRIPT ---
# Change this to whatever image you want to test
test_image = "test_image.png"
generate_comparison(test_image)
