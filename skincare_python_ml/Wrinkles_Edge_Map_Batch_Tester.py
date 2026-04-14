import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import cv2

# --- 1. CONFIGURATION ---
MODEL_PATH  = r'D:\code\Capstone_project\skincare_python_ml\wrinkles_phase1_v2.keras'
TEST_FOLDER = r'E:\dataset_of_capstone\test_images'
THRESHOLD   = 0.70   # ← raised from 0.5 to reduce false positives

print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
except Exception as e:
    print(f"Failed: {e}")
    exit()

if not os.path.exists(TEST_FOLDER):
    print(f"ERROR: Could not find '{TEST_FOLDER}'.")
    exit()

# --- 2. OPENCV FACE DETECTOR ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def crop_face(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )
    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    h_img, w_img = img.shape[:2]
    pad_x = int(w * 0.2)
    pad_y = int(h * 0.2)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    return img[y1:y2, x1:x2]

def convert_to_edge_map(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    face = crop_face(img)
    if face is None:
        print(f"    No face detected — using full image")
        face = img

    face = cv2.resize(face, (224, 224))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.abs(laplacian)

    edge_map = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    edge_map = np.uint8(edge_map)

    # ✅ Mask hair (top 20%) and neck/clothing (bottom 15%)
    # Focuses the model on actual facial skin only
    edge_map[:int(224 * 0.20), :] = 0
    edge_map[int(224 * 0.85):, :] = 0

    return cv2.merge([edge_map, edge_map, edge_map])


# --- 3. DEBUG FOLDER ---
SAVE_DEBUG   = True
DEBUG_FOLDER = os.path.join(TEST_FOLDER, 'debug_edges')
if SAVE_DEBUG:
    os.makedirs(DEBUG_FOLDER, exist_ok=True)

# --- 4. BATCH TESTING LOOP ---
valid_extensions = ('.jpg', '.jpeg', '.png')
image_files      = sorted([
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith(valid_extensions)
    and not f.startswith('wrnk_')     # skip already-processed files
    and not f.startswith('edge_')     # skip debug edge maps
])

print(f"Found {len(image_files)} images.\n")

if len(image_files) == 0:
    print("⚠️  No unprocessed images found.")
    print(f"    Folder contents: {os.listdir(TEST_FOLDER)}")
    exit()

results = []

for serial_number, filename in enumerate(image_files, start=1):
    filepath = os.path.join(TEST_FOLDER, filename)

    try:
        edge_map = convert_to_edge_map(filepath)
        if edge_map is None:
            print(f"  [SKIP] {filename}")
            continue

        if SAVE_DEBUG:
            cv2.imwrite(
                os.path.join(DEBUG_FOLDER, f"edge_{filename}"),
                edge_map
            )

        img_array = np.expand_dims(edge_map.astype(np.float32), axis=0)
        img_array = preprocess_input(img_array)

        raw_score = model.predict(img_array, verbose=0)[0][0]
        label     = "wrinkles_present" if raw_score >= THRESHOLD else "clear_skin"

        ext          = os.path.splitext(filename)[1]
        new_filename = f"wrnk_{serial_number}_{label}_{raw_score:.2f}{ext}"
        new_filepath = os.path.join(TEST_FOLDER, new_filename)
        os.rename(filepath, new_filepath)

        results.append({
            "original": filename,
            "score":    raw_score,
            "label":    label
        })

        print(f"[{serial_number:03d}] {filename:<35} → {label:<20} ({raw_score:.4f})")

    except Exception as e:
        print(f"  [ERROR] {filename}: {e}")

# --- 5. SUMMARY ---
total         = len(results)

if total == 0:
    print("\n⚠️  No images were processed.")
else:
    wrinkle_count = sum(1 for r in results if r['label'] == 'wrinkles_present')
    clear_count   = total - wrinkle_count

    print(f"\n{'='*55}")
    print(f"Total   : {total}")
    print(f"Wrinkles: {wrinkle_count} ({wrinkle_count/total*100:.1f}%)")
    print(f"Clear   : {clear_count}   ({clear_count/total*100:.1f}%)")
    print(f"Threshold: {THRESHOLD}")
    if SAVE_DEBUG:
        print(f"Edge maps: {DEBUG_FOLDER}")

print("\n✅ Done!")