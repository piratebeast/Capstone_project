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
# Load the pre-trained brain, without the final 1000-class guessing layer
base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the brain so it doesn't forget how to see shapes and edges
base_model.trainable = False 

# Attach our custom output node for the Yes/No percentage
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(), 
    layers.Dense(1, activation='sigmoid') # Gives us the 0.0 to 1.0 probability
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
model.save('acne_mvp_model.keras')
print("Model saved as 'acne_mvp_model.keras'. You are ready to build the API gateway!")
