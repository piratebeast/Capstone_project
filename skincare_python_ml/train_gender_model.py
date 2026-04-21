import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import os
from datetime import datetime

# ========== GPU CHECK ==========
print("=" * 60)
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detected: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")
    
if len(gpus) == 0:
    print("\n⚠️  WARNING: No GPU detected! Training will use CPU (very slow)")
    print("Make sure you installed: pip install tensorflow-directml")
else:
    print("\n✅ GPU ready!")
print("=" * 60)

# ========== ENABLE MIXED PRECISION (faster on RX 7600) ==========
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("\n✅ Mixed precision enabled (faster training)")

# ========== CONFIGURATION ==========
DATA_DIR   = r'E:\dataset_of_capstone\gender_dataset'  # ← CHANGE THIS PATH
SAVE_DIR   = r'D:\code\Capstone_project\skincare_python_ml'           # ← CHANGE THIS PATH
IMG_SIZE   = (224, 224)
BATCH_SIZE = 4  # Optimized for 8GB VRAM
AUTOTUNE   = tf.data.AUTOTUNE

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

if not os.path.exists(DATA_DIR):
    print(f"\n❌ ERROR: Could not find '{DATA_DIR}'")
    print("Please update DATA_DIR in the script!")
    exit()

# ========== LOAD DATASET ==========
print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

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
# Expected: ['female', 'male']
# female = label 0, male = label 1

# ========== CLASS WEIGHTS ==========
total_female = len(os.listdir(os.path.join(DATA_DIR, 'female')))
total_male   = len(os.listdir(os.path.join(DATA_DIR, 'male')))
total        = total_female + total_male

weight_for_0 = (1 / total_female) * (total / 2.0)
weight_for_1 = (1 / total_male)   * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"\nDataset statistics:")
print(f"  Female: {total_female:,} images (weight: {weight_for_0:.4f})")
print(f"  Male:   {total_male:,} images (weight: {weight_for_1:.4f})")
print(f"  Total:  {total:,} images")

# ========== DATA AUGMENTATION ==========
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2),
], name='augmentation')

def prepare(ds, augment=False):
    ds = ds.map(lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

train_dataset = prepare(train_dataset, augment=True)
val_dataset   = prepare(val_dataset,   augment=False)

# ========== BUILD MODEL ==========
print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

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
outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)  # Force float32 output

model = tf.keras.Model(inputs, outputs, name='gender_classifier')

print(f"\nModel created:")
print(f"  Total parameters: {model.count_params():,}")
print(f"  Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ========== PHASE 1: TRAIN HEAD ONLY ==========
print("\n" + "=" * 60)
print("PHASE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

phase1_path = os.path.join(SAVE_DIR, 'gender_phase1.keras')
callbacks_phase1 = [
    EarlyStopping(
        monitor='val_auc',
        patience=5,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        phase1_path,
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
]

start_time = datetime.now()
print(f"Started at: {start_time.strftime('%H:%M:%S')}")

history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    class_weight=class_weight,
    callbacks=callbacks_phase1,
    verbose=1
)

phase1_time = datetime.now() - start_time
print(f"\nPhase 1 completed in: {phase1_time}")

# ========== PHASE 2: FINE-TUNE ==========
print("\n" + "=" * 60)
print("PHASE 2: FINE-TUNING LAST RESNET BLOCK")
print("=" * 60)

base_model.trainable = True
for layer in base_model.layers[:143]:
    layer.trainable = False

trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Trainable parameters: {trainable_params:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

phase2_path = os.path.join(SAVE_DIR, 'gender_production.keras')
callbacks_phase2 = [
    EarlyStopping(
        monitor='val_auc',
        patience=7,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        phase2_path,
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

start_time = datetime.now()
print(f"Started at: {start_time.strftime('%H:%M:%S')}")

history2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    class_weight=class_weight,
    callbacks=callbacks_phase2,
    verbose=1
)

phase2_time = datetime.now() - start_time
total_time = phase1_time + phase2_time

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nPhase 1 time: {phase1_time}")
print(f"Phase 2 time: {phase2_time}")
print(f"Total time:   {total_time}")
print(f"\nModels saved:")
print(f"  Phase 1: {phase1_path}")
print(f"  Final:   {phase2_path}")
print("\n✅ Ready for deployment!")

# ========== EVALUATION & VISUAL METRICS ==========
print("\n" + "=" * 60)
print("GENERATING VISUAL METRICS")
print("=" * 60)

print("Running predictions on validation data...")
y_true = []
y_pred_probs = []

# Iterate through the validation dataset to get true labels and predictions
for images, labels in val_dataset:
    y_true.extend(labels.numpy())
    # Predict using the fine-tuned model
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds)

y_true = np.array(y_true).flatten()
y_pred_probs = np.array(y_pred_probs).flatten()

# Apply the 0.5 threshold to get binary 0 or 1 labels
y_pred = (y_pred_probs > 0.5).astype(int)

# 1. Print the Text Report (Accuracy, Precision, Recall, F1-Score)
print("\n--- Classification Report ---")
# Keras alphabetical order: 0 = Female, 1 = Male
target_names = ['Female (0)', 'Male (1)'] 
print(classification_report(y_true, y_pred, target_names=target_names))

# 2. Initialize the visual plotting space (1 row, 3 columns)
plt.figure(figsize=(20, 6))

# --- PLOT A: CONFUSION MATRIX ---
plt.subplot(1, 3, 1)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (Gender)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# --- PLOT B: ROC CURVE (AUC) ---
plt.subplot(1, 3, 2)
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # 50/50 guessing line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# --- PLOT C: TRAINING HISTORY (Loss) ---
# Stitches Phase 1 and Phase 2 history together
plt.subplot(1, 3, 3)
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
epochs_range = range(1, len(loss) + 1)

phase2_start = len(history1.history['loss'])

plt.plot(epochs_range, loss, label='Training Loss', color='blue')
plt.plot(epochs_range, val_loss, label='Validation Loss', color='orange')
plt.axvline(x=phase2_start, color='red', linestyle='--', label='Phase 2 Start')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Optional: Save the figure to your folder
plt.savefig(os.path.join(SAVE_DIR, 'gender_model_metrics.png'), dpi=300)