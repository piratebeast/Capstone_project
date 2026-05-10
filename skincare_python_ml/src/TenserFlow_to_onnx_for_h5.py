import os
# Must be set BEFORE importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import tensorflow as tf
import tf2onnx

model_path = r"D:\code\Capstone_project\skincare_python_ml\gender_keras\gender_production_fixed.h5"
onnx_model_path = model_path.replace(".h5", ".onnx")

print("Loading with Legacy Keras 2...")
model = tf.keras.models.load_model(model_path)

print("Converting to ONNX...")
tf2onnx.convert.from_keras(model, output_path=onnx_model_path, opset=13)

print("✅ Done.")