import json
import os
import shutil
import statistics
import cv2
import numpy as np
import time

# --- Configuration ---
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RAW_DIR      = os.path.join(BASE_DIR, '1_bw_raw_data')
JSON_PATH    = os.path.join(RAW_DIR, 'metadata.json')

WRINKLES_DIR = os.path.join(BASE_DIR, '3_clean_dataset', 'wrinkles_present')
CLEAR_DIR    = os.path.join(BASE_DIR, '3_clean_dataset', 'clear_skin')

os.makedirs(WRINKLES_DIR, exist_ok=True)
os.makedirs(CLEAR_DIR,    exist_ok=True)

# --- Scoring Function ---
def get_wrinkle_score(image_path, landmarks):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    scores = []

    try:
        # Region 1: Forehead
        ly  = int(landmarks[17][1])
        ry  = int(landmarks[26][1])
        lx  = int(landmarks[17][0])
        rx  = int(landmarks[26][0])
        y1  = max(0, min(ly, ry) - 120)
        y2  = min(ly, ry)
        x1, x2 = max(0, lx), min(w, rx)
        forehead = img[y1:y2, x1:x2]
        if forehead.size > 0:
            scores.append(np.mean(forehead))

        # Region 2 & 3: Crow's feet
        for idx in [36, 45]:
            ex = int(landmarks[idx][0])
            ey = int(landmarks[idx][1])
            patch = img[max(0, ey-40):min(h, ey+40),
                        max(0, ex-40):min(w, ex+40)]
            if patch.size > 0:
                scores.append(np.mean(patch))

        # Region 4: Nasolabial folds
        nx = int(landmarks[33][0])
        ny = int(landmarks[33][1])
        patch3 = img[max(0, ny):min(h, ny+80),
                     max(0, nx-60):min(w, nx+60)]
        if patch3.size > 0:
            scores.append(np.mean(patch3))

    except Exception:
        scores.append(np.mean(img[int(h*0.2):int(h*0.5),
                                  int(w*0.2):int(w*0.8)]))

    return float(np.mean(scores)) if scores else None

# --- Load JSON ---
print("Loading metadata.json...")
with open(JSON_PATH, 'r') as f:
    data = json.load(f)
print(f"Total entries: {len(data)}")

# --- Phase 1: Score all images ---
print("\nPhase 1: Scoring images...")
scored     = []
all_scores = []
skipped    = 0
start_time = time.time()

for i, (key, item) in enumerate(data.items()):

    if i % 1000 == 0:
        elapsed = time.time() - start_time
        pct = (i / len(data)) * 100
        print(f"  [{pct:5.1f}%] {i}/{len(data)} images | {elapsed:.1f}s elapsed")

    try:
        # ✅ CORRECT PATH — reads subfolder from file_path in JSON
        # file_path in JSON looks like: "images1024x1024/00000/00000.png"
        # We need:                       RAW_DIR/00000/00000.png

        rel_path  = item['image']['file_path']          # "images1024x1024/00000/00000.png"
        parts     = rel_path.replace('\\', '/').split('/')
        # parts[-1] = "00000.png"  parts[-2] = "00000" (subfolder)
        filename  = parts[-1]                            # "00000.png"
        subfolder = parts[-2]                            # "00000"
        full_path = os.path.join(RAW_DIR, subfolder, filename)

        if not os.path.exists(full_path):
            skipped += 1
            continue

        landmarks = item['image']['face_landmarks']
        score = get_wrinkle_score(full_path, landmarks)

        if score is not None:
            scored.append({
                "filename": filename,
                "subfolder": subfolder,
                "full_path": full_path,
                "score": score
            })
            all_scores.append(score)
        else:
            skipped += 1

    except Exception:
        skipped += 1
        continue

elapsed_total = time.time() - start_time
print(f"\nScoring complete in {elapsed_total:.1f}s")
print(f"Scored  : {len(scored)}")
print(f"Skipped : {skipped}")
print(f"Score range : {min(all_scores):.4f} → {max(all_scores):.4f}")
print(f"Mean score  : {statistics.mean(all_scores):.4f}")

# --- Phase 2: 30/70 Split (NOT median) ---
# Discard ambiguous middle 40% to ensure clean labels
scores_sorted = sorted(all_scores)
n = len(scores_sorted)

LOW_THRESHOLD  = scores_sorted[int(n * 0.30)]  # bottom 30% → clear_skin
HIGH_THRESHOLD = scores_sorted[int(n * 0.70)]  # top 30%    → wrinkles_present

print(f"\n30/70 Split thresholds:")
print(f"  clear_skin cutoff (30th pct)      : {LOW_THRESHOLD:.4f}")
print(f"  wrinkles_present cutoff (70th pct): {HIGH_THRESHOLD:.4f}")
print(f"  Images kept    : ~{int(n*0.60)}")
print(f"  Images discarded (ambiguous middle): ~{int(n*0.40)}")

# --- Phase 3: Sort ---
print("\nPhase 3: Copying images into folders...")

w_count   = 0
c_count   = 0
discarded = 0

for item in scored:
    if item['score'] >= HIGH_THRESHOLD:
        dest = os.path.join(WRINKLES_DIR, item['filename'])
        shutil.copy(item['full_path'], dest)
        w_count += 1

    elif item['score'] <= LOW_THRESHOLD:
        dest = os.path.join(CLEAR_DIR, item['filename'])
        shutil.copy(item['full_path'], dest)
        c_count += 1

    else:
        discarded += 1   # ambiguous middle — skip

print(f"\n✅ Done!")
print(f"wrinkles_present : {w_count}")
print(f"clear_skin       : {c_count}")
print(f"Discarded        : {discarded}")
print(f"\nNow run the audit script on '3_clean_dataset' to verify quality.")