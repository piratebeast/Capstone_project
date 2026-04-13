import json
import os
import base64
import statistics
import cv2
import numpy as np
from pathlib import Path

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WRINKLES_DIR = os.path.join(BASE_DIR, '4_clean_dataset', 'wrinkles_present')
CLEAR_DIR    = os.path.join(BASE_DIR, '4_clean_dataset', 'clear_skin')
REPORT_PATH  = os.path.join(BASE_DIR, 'audit_report.html')

def img_to_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_score(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    img = cv2.resize(img, (256, 256))
    return float(np.mean(img))

# --- Collect all images with scores ---
print("Scanning dataset...")

wrinkle_images = []
clear_images   = []

for fname in os.listdir(WRINKLES_DIR):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        full = os.path.join(WRINKLES_DIR, fname)
        wrinkle_images.append({"path": full, "name": fname, "score": get_score(full)})

for fname in os.listdir(CLEAR_DIR):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        full = os.path.join(CLEAR_DIR, fname)
        clear_images.append({"path": full, "name": fname, "score": get_score(full)})

# Sort by score — lowest scores in wrinkles folder are SUSPICIOUS
wrinkle_images.sort(key=lambda x: x['score'])   # lowest first = most suspicious
clear_images.sort(key=lambda x: x['score'], reverse=True)  # highest first = most suspicious

print(f"wrinkles_present : {len(wrinkle_images)} images")
print(f"clear_skin       : {len(clear_images)} images")

# --- Statistics ---
w_scores = [x['score'] for x in wrinkle_images]
c_scores = [x['score'] for x in clear_images]

print(f"\nwrinkles_present score stats:")
print(f"  Min : {min(w_scores):.4f}  ← check these images manually")
print(f"  Max : {max(w_scores):.4f}")
print(f"  Mean: {statistics.mean(w_scores):.4f}")

print(f"\nclear_skin score stats:")
print(f"  Min : {min(c_scores):.4f}")
print(f"  Max : {max(c_scores):.4f}  ← check these images manually")
print(f"  Mean: {statistics.mean(c_scores):.4f}")

# Overlap warning
overlap = [x for x in wrinkle_images if x['score'] < statistics.mean(c_scores)]
print(f"\n⚠️  Suspicious wrinkle images (score below clear_skin mean): {len(overlap)}")

# --- Build HTML report ---
print("\nBuilding visual audit report...")

def make_grid(images, label, color, limit=50):
    """Show first `limit` images as a visual grid"""
    html = f'<h2 style="color:{color}">{label} — showing {min(limit, len(images))} most suspicious</h2>'
    html += '<div style="display:flex;flex-wrap:wrap;gap:10px;">'
    for item in images[:limit]:
        b64 = img_to_base64(item['path'])
        score_color = "red" if (
            (color == "crimson" and item['score'] < statistics.mean(c_scores)) or
            (color == "green"   and item['score'] > statistics.mean(w_scores))
        ) else "black"
        html += f'''
        <div style="text-align:center;width:160px;">
            <img src="data:image/png;base64,{b64}"
                 style="width:150px;height:150px;object-fit:cover;border:2px solid {score_color}"/>
            <div style="font-size:11px;color:{score_color}">
                {item["name"]}<br/>score: {item["score"]:.3f}
            </div>
        </div>'''
    html += '</div>'
    return html

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }}
        h1   {{ color: #333; }}
        h2   {{ margin-top: 40px; }}
        .stats {{ background: white; padding: 15px; border-radius: 8px;
                  margin: 10px 0; display: inline-block; min-width: 300px; }}
        .warning {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>📊 Dataset Audit Report</h1>

    <div style="display:flex;gap:30px;flex-wrap:wrap;">
        <div class="stats">
            <b>wrinkles_present</b><br/>
            Total images : {len(wrinkle_images)}<br/>
            Mean score   : {statistics.mean(w_scores):.4f}<br/>
            Min score    : {min(w_scores):.4f}<br/>
            Max score    : {max(w_scores):.4f}
        </div>
        <div class="stats">
            <b>clear_skin</b><br/>
            Total images : {len(clear_images)}<br/>
            Mean score   : {statistics.mean(c_scores):.4f}<br/>
            Min score    : {min(c_scores):.4f}<br/>
            Max score    : {max(c_scores):.4f}
        </div>
        <div class="stats">
            <b>Overlap Warning</b><br/>
            <span class="warning">
            Suspicious wrinkle images: {len(overlap)}<br/>
            (score below clear_skin mean)
            </span><br/>
            <small>Red border = likely mislabeled</small>
        </div>
    </div>

    <hr/>
    <!-- MOST SUSPICIOUS WRINKLE IMAGES (lowest scores = smoothest = shouldn't be here) -->
    {make_grid(wrinkle_images, "wrinkles_present — lowest scores first (RED = likely wrong)", "crimson", limit=60)}

    <hr/>
    <!-- MOST SUSPICIOUS CLEAR IMAGES (highest scores = most edges = shouldn't be here) -->
    {make_grid(clear_images, "clear_skin — highest scores first (RED = likely wrong)", "green", limit=60)}

</body>
</html>
"""

# The 'encoding="utf-8"' tells Windows to support all characters, including emojis
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✅ Report saved → open this file in your browser:")
print(f"   {REPORT_PATH}")