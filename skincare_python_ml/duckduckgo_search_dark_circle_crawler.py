import os
import requests
import hashlib
import time
import random
from PIL import Image
from io import BytesIO
from ddgs import DDGS  # Ensure you ran: pip install -U duckduckgo-search

SAVE_DIR = r'E:\dataset_of_capstone\dark_circles_raw'
os.makedirs(SAVE_DIR, exist_ok=True)

queries = [
    "dark circles under eyes close up",
    "severe periorbital hyperpigmentation",
    "under eye dark pigmentation",
    "dark eye bags skin",
    "periorbital melanosis face"
]

def download_images(query, max_images=100): # Reduced count slightly to test stability
    saved = 0
    print(f"\n--- Starting search for: {query} ---")
    
    with DDGS() as ddgs:
        try:
            # The results are now returned as a generator
            results = ddgs.images(
                keywords=query,
                region="wt-wt",
                safesearch="off",
                type_image="photo",
                max_results=max_images
            )

            for r in results:
                if saved >= max_images:
                    break
                
                try:
                    img_url = r.get('image')
                    if not img_url: continue

                    response = requests.get(img_url, timeout=10)
                    if response.status_code != 200: continue

                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    img_hash = hashlib.md5(response.content).hexdigest()
                    filename = os.path.join(SAVE_DIR, f"{img_hash}.jpg")

                    if not os.path.exists(filename):
                        img.save(filename, "JPEG", quality=90)
                        saved += 1
                        print(f"  [{saved}] Saved: {img_hash}.jpg")
                    
                    # Random delay between 1 to 3 seconds per image
                    time.sleep(random.uniform(1.0, 3.0))

                except Exception as e:
                    continue # Silently skip individual image errors

        except Exception as e:
            if "403" in str(e):
                print(f"CRITICAL: Still Rate Limited. Stop the script and wait 20 mins.")
                return False # Signal to stop the whole loop
            print(f"Search error: {e}")
    
    print(f"Done '{query}': {saved} images saved")
    return True

for q in queries:
    success = download_images(q, max_images=100)
    if not success:
        break # Stop if we are being blocked
    # Wait longer between different search queries
    print("Waiting 10 seconds before next query...")
    time.sleep(10)

print("\nProcess finished.")