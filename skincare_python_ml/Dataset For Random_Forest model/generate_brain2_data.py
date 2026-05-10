import pandas as pd
import numpy as np

np.random.seed(42)
NUM_SAMPLES = 3000

# 1. Base Inputs (All 6 from your Brain 1)
df = pd.DataFrame({
    'age': np.random.randint(16, 70, NUM_SAMPLES),
    'gender': np.random.choice(['Male', 'Female'], NUM_SAMPLES, p=[0.4, 0.6])
})

df['wrinkle_score'] = np.clip((df['age'] - 20) / 60.0 + np.random.normal(0, 0.1, NUM_SAMPLES), 0, 1)
df['acne_score'] = np.where(df['age'] < 30, np.clip(np.random.normal(0.6, 0.2, NUM_SAMPLES), 0, 1), np.clip(np.random.normal(0.2, 0.2, NUM_SAMPLES), 0, 1))
df['redness_score'] = np.clip(df['acne_score'] * 0.4 + np.random.normal(0.2, 0.15, NUM_SAMPLES), 0, 1)
df['dark_circle_score'] = np.clip((df['age'] / 100.0) + np.random.normal(0.2, 0.2, NUM_SAMPLES), 0, 1)

# Added Dark Spots!
df['dark_spot_score'] = np.clip((df['age'] / 80.0) + (df['acne_score'] * 0.3) + np.random.normal(0.1, 0.15, NUM_SAMPLES), 0, 1)

# 2. Complex Clinical Logic (The Multi-Label Targets)
# This isn't just an if/else anymore. It's evaluating competing scores.
def calculate_treatments(row):
    # Default to 0 (No)
    treatments = {
        'needs_salicylic_acid': 0, # For Acne
        'needs_retinol': 0,        # For Wrinkles
        'needs_vitamin_c': 0,      # For Pigmentation (Spots/Circles)
        'needs_niacinamide': 0,    # Good all-rounder
        'needs_azelaic_acid': 0    # For Acne + Redness (Sensitive)
    }
    
    # Complex rules that the Random Forest will have to learn
    if row['acne_score'] > 0.5:
        if row['redness_score'] < 0.6: 
            treatments['needs_salicylic_acid'] = 1
        else:
            treatments['needs_azelaic_acid'] = 1 # Too red for Salicylic
            
    if row['wrinkle_score'] > 0.5 and row['redness_score'] < 0.7:
        treatments['needs_retinol'] = 1
        
    if row['dark_spot_score'] > 0.4 or row['dark_circle_score'] > 0.6:
        treatments['needs_vitamin_c'] = 1
        
    if row['acne_score'] > 0.3 and row['dark_spot_score'] > 0.3:
        treatments['needs_niacinamide'] = 1
        
    return pd.Series(treatments)

# Apply the logic to create multiple target columns
targets = df.apply(calculate_treatments, axis=1)
df = pd.concat([df, targets], axis=1)

df.to_csv('advanced_skincare_patients.csv', index=False)
print("✅ Advanced Multi-Label Dataset Generated!")