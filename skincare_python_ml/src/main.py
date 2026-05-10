from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# Import your new master pipeline from the adjacent predictor.py file
from predictor import analyze_face_pipeline

app = FastAPI(title="Dual-Brain Skincare API")

@app.post("/predict")
async def analyze_face(file: UploadFile = File(...)):
    try:
        # Read the raw bytes sent from your C# HttpClient
        image_bytes = await file.read()
        
        # Send bytes to predictor.py, get the final JSON dictionary back
        # Note: You can pass age from your C# app later. Defaulting to 25 here.
        final_payload = analyze_face_pipeline(image_bytes, user_age=25)
        
        return final_payload

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)