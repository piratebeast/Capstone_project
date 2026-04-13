import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input # CRITICAL IMPORT
import numpy as np
import os

# --- 1. CONFIGURATION ---
# Replace with the exact absolute path to where your model is saved
MODEL_PATH = r'D:\code\Capstone_project\skincare_python_ml\wrinkles_v2_production.keras' 
TEST_FOLDER = r'E:\dataset_of_capstone\test_images'

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(TEST_FOLDER):
    print(f"ERROR: Could not find the folder '{TEST_FOLDER}'. Please create it and add images.")
    exit()

valid_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(valid_extensions)]

# Sort files to ensure consistent serial numbering
image_files.sort() 

print(f"Found {len(image_files)} images to test. Starting batch process...\n")

# --- 2. AUTOMATED TESTING LOOP ---
# Using enumerate replaces the inefficient new_func() completely
for serial_number, filename in enumerate(image_files, start=1):
    old_filepath = os.path.join(TEST_FOLDER, filename)
    
    try:
        # Load and process the image
        img = image.load_img(old_filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # --- 3. CRITICAL DATA PREPROCESSING ---
        # Unless a specific Rescaling or preprocessing layer was built directly 
        # into the top of the model architecture during training, 
        # the array must be preprocessed to match the pre-trained weights.
        img_array = preprocess_input(img_array) 
        
        # Make the prediction
        prediction = model.predict(img_array, verbose=0)
        
        # The final Dense(1, sigmoid) layer outputs a probability [0.0, 1.0]
        raw_score = prediction[0][0] 
        
        # --- 4. RENAME THE FILE ---
        ext = os.path.splitext(filename)[1]
        
        # Formats output to look like: v4.1_1_0.87.jpg
        new_filename = f"v4.1_{serial_number}_{raw_score:.2f}{ext}"
        new_filepath = os.path.join(TEST_FOLDER, new_filename)
        
        os.rename(old_filepath, new_filepath)
        print(f"Analyzed & Renamed: {filename}  -->  {new_filename}")
        
    except Exception as e:
        print(f"Failed to process {filename}: {e}")

print("\nBatch testing complete! Open your folder to see the results.")