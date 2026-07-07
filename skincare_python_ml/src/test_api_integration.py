import requests
import json

def test_fastapi_endpoint():
    print("=" * 70)
    print("🚀 TESTING FASTAPI PAYLOAD INTEGRATION CONTRACT")
    print("=" * 70)

    # 1. Configuration parameters
    url = "http://127.0.0.1:8000/analyze"
    image_path = r"D:\code\Capstone_project\skincare_python_ml\test.jpg"
    
    # Mimic the form data sent by ASP.NET Core
    form_data = {
        "user_age": 24  # Sending a mock age value
    }

    try:
        # 2. Open your real test image file as binary stream data
        print("[1/3] Reading target test image file binary stream...")
        with open(image_path, "rb") as f:
            files = {"file": ("test.jpg", f, "image/jpeg")}
            
            print("[2/3] Broadcasting POST request to FastAPI server context...")
            response = requests.post(url, data=form_data, files=files)
            
        # 3. Parse and validate the incoming response data payload
        print("[3/3] Analyzing server response contract...")
        if response.status_code != 200:
            print(f"❌ Server Error! Status Code: {response.status_code}")
            print(f"   Details: {response.text}")
            return

        payload = response.json()
        print("\n✅ SUCCESS! FastAPI responded with a clean 200 OK code.\n")
        
        # 4. Verify structural keys exist matching your data architecture
        print("-" * 50)
        print(f"📅 Scan Date Matrix: {payload.get('scanDate')}")
        print(f"🧠 Assigned Routine Class: {payload.get('routineClass')}")
        print(f"🎯 Inference Confidence: {payload.get('confidence')}")
        print("-" * 50)
        
        print("\n📈 Diagnostics Severity Values Check:")
        for key, val in payload.get('diagnostics', {}).items():
            print(f"   -> {key}: {val}%")
            
        print("\n🗺️ Matrix Heatmap Arrays Array-Lengths Validation:")
        heatmaps = payload.get('heatmaps', {})
        for condition, array_data in heatmaps.items():
            print(f"   -> {condition}: Found flat array of {len(array_data)} floating elements.")
            
            # Catching a structural payload defect check
            if len(array_data) != 50176:
                print(f"      ⚠️ WARNING: Matrix array size is {len(array_data)}, expected 50176!")

        print("-" * 50)
        print("🏁 API HEALTH CHECK COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"❌ Connection Failed: Is your FastAPI server currently running? \n   Details: {str(e)}")
        print("=" * 70)

if __name__ == "__main__":
    test_fastapi_endpoint()