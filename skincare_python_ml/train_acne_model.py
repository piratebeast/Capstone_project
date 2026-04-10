import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import os

# --- 1. SETUP & CONFIGURATION ---
print("TensorFlow Version:", tf.__version__)
print("Checking for folders...")

# Make sure this matches the exact name of your main folder
DATA_DIR = r'E:\dataset_of_capstone\acne_dataset' 

if not os.path.exists(DATA_DIR):
    print(f"ERROR: Could not find the folder '{DATA_DIR}'. Make sure it is in the same directory as this script!")
    exit()

# ResNet50 requires exactly 224x224 images
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32

# --- 2. LOAD THE IMAGES ---
print("\nLoading Training Data (80%)...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("\nLoading Validation Data (20%)...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# --- 3. BUILD THE RESNET50 ARCHITECTURE ---
print("\nDownloading and building ResNet50 model...")
base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

# 1. Create the Augmentation Layer
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1), # Rotates up to 10%
  layers.RandomZoom(0.2),     # Zooms in up to 20%
])

# 2. Add it to your model pipeline BEFORE the base_model
model = models.Sequential([
    data_augmentation,        # Scrambles the image first
    base_model,               # Extracts features
    layers.GlobalAveragePooling2D(), 
    layers.Dense(1, activation='sigmoid') 
])

# --- 4. COMPILE AND TRAIN ---
print("\nCompiling model...")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n=== STARTING TRAINING ===")
# We will start with just 10 epochs for the MVP
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# --- 5. SAVE THE FINISHED MODEL ---
print("\nTraining complete! Saving the model for the API...")
model.save('acne_v2_model.keras')
print("Model saved as 'acne_v2_model.keras'. You are ready to build the API gateway!")
