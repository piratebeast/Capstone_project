import cv2
import os

# --- PATHS (Adjust these to your exact UTKFace location) ---
INPUT_DIR = r'E:\archive\UTKFace' 
OUTPUT_DIR = r'E:\dataset_of_capstone\eye_crops_mining'

# Load the detector
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Starting extraction... This may take a while for 20k images.")

count = 0
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img = cv2.imread(os.path.join(INPUT_DIR, filename))
    if img is None: continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    
    # We need exactly 2 eyes to make a proper "Eye-Box"
    if len(eyes) == 2:
        # Sort by X-coordinate so eyes[0] is Left and eyes[1] is Right
        eyes = sorted(eyes, key=lambda e: e[0])
        x1, y1, w1, h1 = eyes[0]
        x2, y2, w2, h2 = eyes[1]
        
        # Calculate the bounding box for the EYE AREA
        # We go slightly above the eyes and significantly BELOW the eyes
        top = min(y1, y2) - int(h1 * 0.4)
        bottom = max(y1 + h1, y2 + h2) + int(h1 * 1.3) # 1.3 adds room for the circles
        left = x1 - int(w1 * 0.5)
        right = x2 + w2 + int(w2 * 0.5)
        
        # Crop and check if valid
        crop = img[max(0, top):bottom, max(0, left):right]
        
        if crop.size > 0:
            # Resize to a consistent width for easier manual sorting later
            # (Keeping aspect ratio is better for now)
            target_width = 400
            ratio = target_width / crop.shape[1]
            dim = (target_width, int(crop.shape[0] * ratio))
            resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"eye_{filename}"), resized)
            count += 1

    # Progress tracker
    if count % 100 == 0:
        print(f"Processed {count} eye-crops...")

print(f"✅ Done! Extracted {count} eye-regions to {OUTPUT_DIR}")