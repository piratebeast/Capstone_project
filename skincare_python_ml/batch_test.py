import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = 'acne_v3_production.keras'
TEST_FOLDER = r'E:\dataset_of_capstone\test_images' # The folder where you put your test photos

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Check if the test folder exists
if not os.path.exists(TEST_FOLDER):
    print(f"ERROR: Could not find the folder '{TEST_FOLDER}'. Please create it and add images.")
    exit()

# Find all the images in the folder (ignoring other files like .txt)
valid_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(valid_extensions)]

print(f"Found {len(image_files)} images to test. Starting batch process...\n")

# --- 2. AUTOMATED TESTING LOOP ---
for filename in image_files:
    # Get the exact location of the current file
    old_filepath = os.path.join(TEST_FOLDER, filename)
    
    try:
        # Load and process the image
        img = image.load_img(old_filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make the prediction (verbose=0 stops it from printing the progress bar every time)
        prediction = model.predict(img_array, verbose=0)
        raw_score = prediction[0][0]
        
        # --- 3. RENAME THE FILE ---
        # Extract the original extension (e.g., '.jpg')
        ext = os.path.splitext(filename)[1]
        
        # Create the new filename: "v1_0.2503.jpg"
        new_filename = f"v1_{raw_score:.4f}{ext}"
        new_filepath = os.path.join(TEST_FOLDER, new_filename)
        
        # Rename the physical file on the hard drive
        os.rename(old_filepath, new_filepath)
        
        # Print a clean log to the console
        print(f"Analyzed & Renamed: {filename}  -->  {new_filename}")
        
    except Exception as e:
        print(f"Failed to process {filename}: {e}")

print("\nBatch testing complete! Open your folder to see the results.")