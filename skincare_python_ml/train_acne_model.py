import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input  # ← CRITICAL FIX
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os

# --- 1. SETUP & CONFIGURATION ---
print("TensorFlow Version:", tf.__version__)

DATA_DIR = '/content/acne_dataset'
if not os.path.exists(DATA_DIR):
    print(f"ERROR: Could not find '{DATA_DIR}'.")
    exit()

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE  # ← Speeds up data loading

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

# Sanity check — make sure your folders are named correctly
class_names = train_dataset.class_names
print(f"\nDetected classes: {class_names}")
# Expected output: ['acne', 'no_acne']  ← folder names must match this

# --- 3. HANDLE CLASS IMBALANCE ---
# Count images per class to compute weights
total_acne = len(os.listdir(os.path.join(DATA_DIR, class_names[0])))
total_no_acne = len(os.listdir(os.path.join(DATA_DIR, class_names[1])))
total = total_acne + total_no_acne

weight_for_0 = (1 / total_acne) * (total / 2.0)
weight_for_1 = (1 / total_no_acne) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"\nClass weights: {class_weight}")
# If both are 1.0, your dataset is balanced — great!

# --- 4. BUILD OPTIMIZED DATA PIPELINE ---
# Apply ResNet50's required preprocessing BEFORE training
def prepare(ds, augment=False):
    # Apply ResNet50 preprocessing (scales pixels correctly)
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    if augment:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
            layers.RandomBrightness(0.1),  # ← helpful for varied lighting conditions
            layers.RandomContrast(0.1),    # ← helps with different skin tones
        ])
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(buffer_size=AUTOTUNE)  # ← loads next batch while GPU trains

train_dataset = prepare(train_dataset, augment=True)
validation_dataset = prepare(validation_dataset, augment=False)

# --- 5. BUILD THE MODEL ---
print("\nBuilding ResNet50 model...")
base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze for Phase 1

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)          # ← Prevents overfitting
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# --- 6. PHASE 1: TRAIN ONLY THE HEAD ---
print("\n=== PHASE 1: Training classification head ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]  # AUC is better for imbalanced data
)

callbacks_phase1 = [
    EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True,
                  mode='max', verbose=1),
    ModelCheckpoint('/content/drive/MyDrive/acne_phase1.keras',
                    monitor='val_auc', save_best_only=True, mode='max', verbose=1),
]

history1 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    class_weight=class_weight,  # ← handles imbalance
    callbacks=callbacks_phase1
)

# --- 7. PHASE 2: FINE-TUNING (Unfreeze last ResNet block) ---
print("\n=== PHASE 2: Fine-tuning last ResNet block ===")

# Unfreeze from layer 143 onward (last conv block of ResNet50)
base_model.trainable = True
for layer in base_model.layers[:143]:
    layer.trainable = False

# Use a much lower learning rate to avoid destroying pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # ← 100x smaller
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

callbacks_phase2 = [
    EarlyStopping(monitor='val_auc', patience=7, restore_best_weights=True,
                  mode='max', verbose=1),
    ModelCheckpoint('/content/drive/MyDrive/acne_v3_production.keras',
                    monitor='val_auc', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history2 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,
    class_weight=class_weight,
    callbacks=callbacks_phase2
)

print("\nTraining complete! Final model saved to Google Drive.")