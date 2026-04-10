from PIL import Image
import os

# Set your folder paths (Make sure these match your actual D: or E: drive paths)
input_folder = r'E:\dataset_of_capstone\acne_dataset\clear_skin'
output_folder = r'E:\dataset_of_capstone\acne_dataset\clear_skin_cropped'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Starting to crop images...")
counter = 0

# Loop through every image in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Calculate the crop box (Grabbing the center 50% of the image)
                left = width * 0.25
                top = height * 0.30  # Start slightly lower to avoid hair/forehead
                right = width * 0.75
                bottom = height * 0.80 # End before the neck/shoulders
                
                # Crop and save
                cropped_img = img.crop((left, top, right, bottom))
                save_path = os.path.join(output_folder, filename)
                cropped_img.save(save_path)
                
                counter += 1
                if counter % 100 == 0:
                    print(f"Cropped {counter} images...")
                    
        except Exception as e:
            print(f"Error cropping {filename}: {e}")

print(f"\nDone! Successfully cropped {counter} images.")
print(f"You can now replace your old 'clear_skin' folder with 'clear_skin_cropped'.")