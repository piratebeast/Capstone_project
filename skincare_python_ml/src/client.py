import requests
import json

# The URL where your FastAPI server is running
API_URL = "http://localhost:8000/predict"

# Path to a test image on your computer
# (Make sure this image actually exists in the same folder!)
TEST_IMAGE_PATH = "test_face.jpg" 

def test_pipeline():
    print(f"Sending {TEST_IMAGE_PATH} to {API_URL}...")
    
    try:
        # Open the image file in binary read mode ('rb')
        with open(TEST_IMAGE_PATH, 'rb') as img_file:
            # Create the multipart form data payload
            files = {'file': (TEST_IMAGE_PATH, img_file, 'image/jpeg')}
            
            # Send the POST request to FastAPI
            response = requests.post(API_URL, files=files)
            
            # Check if the request was successful (HTTP 200)
            if response.status_code == 200:
                print("\n✅ API Success! Received Payload:\n")
                
                # Parse and print the pretty JSON response
                result = response.json()
                print(json.dumps(result, indent=4))
                
            else:
                print(f"❌ Error: API returned status code {response.status_code}")
                print(response.text)
                
    except FileNotFoundError:
        print(f"❌ Error: Could not find the image '{TEST_IMAGE_PATH}'.")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API. Is FastAPI running?")

if __name__ == "__main__":
    test_pipeline()