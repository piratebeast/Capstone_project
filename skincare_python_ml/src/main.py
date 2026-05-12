import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uvicorn

# Import your new master pipeline from the adjacent predictor.py file
from predictor import analyze_face_pipeline

app = FastAPI(title="Dual-Brain Skincare API")

# --- INITIALIZE MEDIAPIPE ONCE AT STARTUP ---
mp_face_detection = mp.solutions.face_detection
# model_selection=0 is optimized for close-range faces (like selfies within 2 meters)
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

@app.post("/analyze") 
async def analyze_face(
    file: UploadFile = File(...),
    user_age: int = Form(25) # <-- NEW: Accepts age from C#, defaults to 25 if missing
):
    try:
        # 1. Read the raw bytes sent from your C# HttpClient
        image_bytes = await file.read()
        
        # ==========================================
        # 2. VALIDATION PHASE (MediaPipe)
        # ==========================================
        # Convert bytes into an OpenCV-readable format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image format or corrupted file.")

        # Convert to RGB (MediaPipe requires RGB, OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run the blazing fast face detection
        results = face_detector.process(img_rgb)

        # Enforce the rules: Must have exactly 1 face
        if not results.detections:
            raise HTTPException(
                status_code=400, 
                detail="NO_FACE_DETECTED" 
            )
            
        if len(results.detections) > 1:
            raise HTTPException(
                status_code=400, 
                detail="MULTIPLE_FACES_DETECTED"
            )
        # ==========================================

        # 3. PREDICTION PHASE
        # Now passing the REAL age that was calculated by your C# backend!
        final_payload = analyze_face_pipeline(image_bytes, user_age=user_age)
        
        return final_payload

    except HTTPException:
        # If we explicitly raised an HTTPException (like the face errors), let it pass through
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)