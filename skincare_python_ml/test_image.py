import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Load your trained MVP model
print("Loading model...")
model = tf.keras.models.load_model('acne_v2_model.keras')

# 2. Load the single user image (Just like your API will do!)
image_path = 'test_photo.jpg' 

img = image.load_img(image_path, target_size=(224, 224)) # Resize for ResNet50

# 3. Convert the image to a mathematical array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) # Keras expects a "batch", so we make a batch of 1

# 4. Make the prediction!
print("Analyzing skin...")
prediction = model.predict(img_array)
raw_score = prediction[0][0]

# 5. Format the output
print("\n--- RESULTS ---")
print(f"Raw Probability: {raw_score:.4f}")

# Remember: closer to 0 meant Kaggle (Acne), closer to 1 meant CelebA (Clear)
if raw_score < 0.5:
    print("Prediction: ACNE DETECTED")
else:
    print("Prediction: CLEAR SKIN")