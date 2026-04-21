import os
import shutil

UTKFACE_DIR = r'E:\archive\UTKFace'  # ← Update this
OUTPUT_DIR  = r'E:\dataset_of_capstone\gender_dataset'  # ← Update this
MIN_AGE = 13  # ← Exclude pre-pubescent children

os.makedirs(f'{OUTPUT_DIR}/female', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/male', exist_ok=True)

female_count = 0
male_count = 0
skipped_age = 0
skipped_other = 0

print(f"Organizing UTKFace dataset (age ≥ {MIN_AGE})...\n")

for filename in os.listdir(UTKFACE_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    try:
        parts = filename.split('_')
        if len(parts) < 2:
            skipped_other += 1
            continue
        
        age = int(parts[0])
        gender = int(parts[1])
        
        # Skip pre-pubescent children
        if age < MIN_AGE:
            skipped_age += 1
            continue
        
        source_path = os.path.join(UTKFACE_DIR, filename)
        
        if gender == 0:  # Female
            dest_path = os.path.join(OUTPUT_DIR, 'female', filename)
            shutil.copy2(source_path, dest_path)
            female_count += 1
            
        elif gender == 1:  # Male
            dest_path = os.path.join(OUTPUT_DIR, 'male', filename)
            shutil.copy2(source_path, dest_path)
            male_count += 1
        else:
            skipped_other += 1
            
        if (female_count + male_count) % 1000 == 0:
            print(f"Processed: {female_count + male_count:,} images...")
            
    except Exception as e:
        skipped_other += 1
        continue

print(f"\n{'='*60}")
print(f"✅ Dataset organized successfully!")
print(f"{'='*60}")
print(f"Female: {female_count:,} images")
print(f"Male:   {male_count:,} images")
print(f"Skipped (age < {MIN_AGE}): {skipped_age:,}")
print(f"Skipped (parsing errors): {skipped_other:,}")
print(f"\nDataset ready at: {OUTPUT_DIR}")