import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# --- 1. SETUP & CONFIGURATION ---
print("TensorFlow Version:", tf.__version__)
print("Checking for folders...")

# Updated to your exact absolute path
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

# RE-ADDED: Data Augmentation to prevent overfitting over 50 epochs!
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.2),
])

# Attach our custom output node for the Yes/No percentage
model = models.Sequential([
    data_augmentation,        # Scrambles the image slightly
    base_model,               # Extracts features
    layers.GlobalAveragePooling2D(), 
    layers.Dense(1, activation='sigmoid') 
])

# --- 4. COMPILE AND TRAIN ---
print("\nCompiling model...")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- 5. PROFESSIONAL CALLBACKS ---
# 1. Stop early if the model stops learning for 5 epochs straight
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# 2. Only save the model when it hits a new accuracy high score
checkpoint = ModelCheckpoint('acne_v3_production.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

print("\n=== STARTING TRAINING ===")
# We set it to 50, but EarlyStopping might stop it at epoch 15 or 20!
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=50, 
    callbacks=[early_stop, checkpoint]
)

print("\nTraining complete! The best model was automatically saved as 'acne_v3_production.keras'.")
do i need to change any code here