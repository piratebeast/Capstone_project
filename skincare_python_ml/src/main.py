from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

app = FastAPI(title="Dual-Brain Skincare API")

# Load the standard OpenCV face detection model
# (This comes pre-installed with the opencv-python package)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_image(image_bytes: bytes):
    # 1. Decode the image from the C# Multipart Form request
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # OpenCV loads images in BGR format
    
    if img is None:
        raise ValueError("Could not decode the image file.")

    # Convert BGR to standard RGB 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- STEP 1: FACIAL CROPPING ---
    # Convert to grayscale just for the detection algorithm
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) > 0:
        # Grab the first detected face (x, y, width, height)
        x, y, w, h = faces[0]
        face_crop = img_rgb[y:y+h, x:x+w]
    else:
        # Fallback: If no face is found, use the whole image so the API doesn't crash
        face_crop = img_rgb

    # --- STEP 2: RESIZING ---
    # Strictly enforce the 224x224 target size using Bilinear Interpolation
    resized_img = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Convert to float32 so we can do math on the pixels
    img_float = np.array(resized_img, dtype=np.float32)

    # --- STEP 3: THE PREPROCESSING SPLIT (CRITICAL) ---
    
    # PATH A: The Legacy Acne Model
    # Math: Divide by 255.0 to squish pixels between 0.0 and 1.0
    img_path_a = img_float / 255.0
    tensor_path_a = np.expand_dims(img_path_a, axis=0) # Shape: (1, 224, 224, 3)

    # PATH B: The ResNet50 Models
    # Math: Keras preprocess_input (Flips RGB to BGR, subtracts ImageNet means)
    img_path_b = np.copy(img_float)
    tensor_path_b = np.expand_dims(img_path_b, axis=0)
    tensor_path_b = preprocess_input(tensor_path_b) # Shape: (1, 224, 224, 3)

    return tensor_path_a, tensor_path_b


@app.post("/predict")
async def analyze_face(file: UploadFile = File(...)):
    try:
        # Read the raw bytes sent from your C# HttpClient
        image_bytes = await file.read()
        
        # Run your custom preprocessing pipeline
        tensor_path_a, tensor_path_b = process_image(image_bytes)
        
        # ---------------------------------------------------------
        # TODO: Replace these mock models with your actual .predict() functions
        # acne_score = legacy_acne_model.predict(tensor_path_a)[0][0]
        # resnet_scores = resnet_models.predict(tensor_path_b)
        # ---------------------------------------------------------

        # MOCK BRAIN 1 OUTPUT (Simulating CNN Results)
        acne_val = 82.5
        dark_spots_val = 60.2
        wrinkles_val = 15.0
        redness_val = 45.1
        dark_circles_val = 30.0
        gender_val = "Male" # From ResNet classifier

        # MOCK BRAIN 2 OUTPUT (Simulating Random Forest Results)
        # This JSON perfectly matches the snake_case Contract your C# DTO is expecting!
        return {
            "diagnostics": {
                "acne": acne_val,
                "dark_spots": dark_spots_val,
                "wrinkles": wrinkles_val,
                "redness": redness_val,
                "dark_circles": dark_circles_val,
                "gender": gender_val
            },
            "routine_class": "ANTI_AGING_SENSITIVE",
            "confidence": 0.92,
            "regimen_schedule": {
                "daily_am": [
                    {"step": 1, "product": "Gentle Milk Cleanser", "purpose": "Cleanse without stripping"},
                    {"step": 2, "product": "Centella Asiatica Serum", "purpose": "Soothe morning redness"},
                    {"step": 3, "product": "Mineral SPF 50+", "purpose": "Crucial UV protection"}
                ],
                "daily_pm": [
                    {"step": 1, "product": "Oil Cleanser", "purpose": "Break down SPF/Makeup"},
                    {"step": 2, "product": "Ceramide Moisturizer", "purpose": "Barrier repair"}
                ],
                "weekly_treatments": [
                    {
                        "product": "Encapsulated Retinol (0.2%)", 
                        "frequency": "2x a week (e.g., Tuesday/Friday PM)",
                        "instructions": "Apply pea-sized amount after moisturizer to reduce irritation."
                    }
                ]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Runs the server on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)