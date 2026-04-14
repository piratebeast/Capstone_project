import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os

print("TensorFlow Version:", tf.__version__)

# --- 1. CONFIGURATION ---
DATA_DIR   = '/content/redness_dataset'
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

if not os.path.exists(DATA_DIR):
    print(f"ERROR: Could not find '{DATA_DIR}'.")
    exit()

# --- 2. LOAD DATASET ---
print("\nLoading training data (80%)...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode='binary',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Loading validation data (20%)...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode='binary',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Confirm class order
class_names = train_dataset.class_names
print(f"\nDetected classes: {class_names}")
# Expected: ['clear_skin', 'redness_present']
# clear_skin = label 0, redness_present = label 1

# --- 3. CLASS IMBALANCE WEIGHTS ---
total_clear    = len(os.listdir(os.path.join(DATA_DIR, 'clear_skin')))
total_redness = len(os.listdir(os.path.join(DATA_DIR, 'redness_present')))
total          = total_clear + total_redness

weight_for_0 = (1 / total_clear)    * (total / 2.0)
weight_for_1 = (1 / total_redness) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"\nclear_skin       : {total_clear} images")
print(f"redness_present : {total_redness} images")
print(f"Class weights    : {class_weight}")

# --- 4. DATA PIPELINE ---
# NEW — add brightness/contrast since these are real color photos now
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2),   # ← handles different lighting
    layers.RandomContrast(0.2),     # ← handles different skin tones
])

def prepare(ds, augment=False):
    ds = ds.map(lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

train_dataset = prepare(train_dataset, augment=True)
val_dataset   = prepare(val_dataset,   augment=False)

# --- 5. BUILD MODEL ---
print("\nBuilding ResNet50 model...")
base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(224, 224, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# --- 6. PHASE 1: TRAIN HEAD ONLY ---
print("\n=== PHASE 1: Training classification head ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

callbacks_phase1 = [
    EarlyStopping(
        monitor='val_auc',
        patience=5,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        '/content/drive/MyDrive/redness_phase1.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
]

history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    class_weight=class_weight,
    callbacks=callbacks_phase1
)

# --- 7. PHASE 2: FINE-TUNE LAST RESNET BLOCK ---
print("\n=== PHASE 2: Fine-tuning last ResNet block ===")
base_model.trainable = True
for layer in base_model.layers[:143]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_auc',
        patience=7,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        '/content/drive/MyDrive/redness_v1_production.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

history2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    class_weight=class_weight,
    callbacks=callbacks_phase2
)

print("\n✅ Training complete!")
print("Phase 1 model → redness_phase1.keras")
print("Final model   → redness_v1_production.keras")