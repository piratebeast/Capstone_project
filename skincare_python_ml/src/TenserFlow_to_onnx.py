import tensorflow as tf
import subprocess
import sys
import os
import shutil

# 1. The model you want to convert
model_path = r"D:\code\Capstone_project\skincare_python_ml\gender_keras\gender_production_fixed.h5"
onnx_model_path = model_path.replace(".h5", ".onnx")
saved_model_dir = "temp_saved_model_dir"

print(f"Loading Keras 3 model: {model_path}...")
model = tf.keras.models.load_model(model_path)

# 2. Export to the standard TensorFlow SavedModel format
# This bypasses the Keras 3 internal tensor naming issue
print("Exporting to SavedModel format...")
model.export(saved_model_dir)

# 3. Call tf2onnx via command line to convert the SavedModel
print(f"Converting SavedModel to ONNX: {onnx_model_path}...")
command = [
    sys.executable, "-m", "tf2onnx.convert",
    "--saved-model", saved_model_dir,
    "--output", onnx_model_path,
    "--opset", "13"
]

try:
    # Run the command and wait for it to finish
    subprocess.run(command, check=True)
    print(f"✅ Successfully converted to {onnx_model_path}")
except subprocess.CalledProcessError as e:
    print(f"❌ Conversion failed: {e}")
finally:
    # Clean up the temporary SavedModel folder
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)