import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn

# Import your master pipeline from the adjacent predictor.py file
from predictor import analyze_face_pipeline

app = FastAPI(title="Dual-Brain Skincare API")

# --- INITIALIZE NEW MEDIAPIPE TASKS API ONCE AT STARTUP ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = r"D:\code\Capstone_project\skincare_python_ml\models\blaze_face_short_range.tflite"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)
# Persistent global tasks detector
face_detector = vision.FaceDetector.create_from_options(options)

@app.post("/analyze") 
async def analyze_face(
    file: UploadFile = File(...),
    user_age: int = Form(25) # Accepts age from C#, defaults to 25 if missing
):
    try:
        # 1. Read the raw bytes sent from your C# HttpClient
        image_bytes = await file.read()
        
        # ==========================================
        # 2. VALIDATION PHASE (Modern MediaPipe Tasks)
        # ==========================================
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image format or corrupted file.")

        # MediaPipe Tasks require RGB channel formatting
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert OpenCV matrix frame to a formal MediaPipe Image object wrapper
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Run modern face detection pass
        detection_result = face_detector.detect(mp_image)

        # Enforce the system rules: Must have exactly 1 face
        if not detection_result.detections:
            raise HTTPException(
                status_code=400, 
                detail="NO_FACE_DETECTED" 
            )
            
        if len(detection_result.detections) > 1:
            raise HTTPException(
                status_code=400, 
                detail="MULTIPLE_FACES_DETECTED"
            )
        # ==========================================

        # 3. PREDICTION PHASE
        final_payload = analyze_face_pipeline(image_bytes, user_age=user_age)
        
        return final_payload

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)