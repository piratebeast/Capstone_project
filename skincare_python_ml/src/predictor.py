import cv2
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input


# ---------------------------------------------------------------------------
# 1. LOAD MODELS  (global scope — loads once on FastAPI startup)
# ---------------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

try:
    model_path   = r"D:\code\Capstone_project\skincare_python_ml\brain2_random_forest.pkl"
    brain2_artifact = joblib.load(model_path)

    rf_model    = brain2_artifact['model']
    rf_features = brain2_artifact['features']   # exact feature order
    rf_targets  = brain2_artifact['targets']    # exact target order
    print("✅ Brain 2 loaded successfully.")
except Exception as e:
    print(f"❌ Brain 2 load failed: {e}")
    rf_model = rf_features = rf_targets = None

# Load Brain 1 ONNX models
try:
    providers = ['CPUExecutionProvider']
    onnx_base = r"D:\code\Capstone_project\skincare_python_ml"
 
    session_acne        = ort.InferenceSession(onnx_base + r"\acne_keras\acne_mvp_model.onnx",             providers=providers)
    session_dark_spots  = ort.InferenceSession(onnx_base + r"\dark_spots_keras\dark_spots_phase1.onnx",    providers=providers)
    session_wrinkles    = ort.InferenceSession(onnx_base + r"\wrinkles_keras\wrinkles_v2_production.onnx", providers=providers)
    session_redness     = ort.InferenceSession(onnx_base + r"\redness_keras\redness_v1_production.onnx",   providers=providers)
    session_dark_circle = ort.InferenceSession(onnx_base + r"\dark_circle_keras\dark_circle_final.onnx",   providers=providers)
    session_gender      = ort.InferenceSession(onnx_base + r"\gender_keras\gender_production_fixed.onnx",  providers=providers)
 
    # Cache input names for prediction mapping
    in_acne        = session_acne.get_inputs()[0].name
    in_dark_spots  = session_dark_spots.get_inputs()[0].name
    in_wrinkles    = session_wrinkles.get_inputs()[0].name
    in_redness     = session_redness.get_inputs()[0].name
    in_dark_circle = session_dark_circle.get_inputs()[0].name
    in_gender      = session_gender.get_inputs()[0].name
 
    print("✅ Brain 1 (ONNX) loaded successfully.")
except Exception as e:
    print(f"❌ Brain 1 ONNX load failed: {e}")
    session_acne = session_dark_spots = session_wrinkles = None
    session_redness = session_dark_circle = session_gender = None

# ---------------------------------------------------------------------------
# 2. IMAGE PROCESSING
# ---------------------------------------------------------------------------
def process_image(image_bytes: bytes):
    """
    Decodes image bytes, detects the primary face, crops and resizes to 224×224.
    Returns two tensors:
      tensor_path_a — normalised [0,1]  (for custom CNN heads)
      tensor_path_b — ResNet50 preprocess_input  (for Brain 1 ResNet50)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure the upload is a valid JPEG/PNG.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )
    face_crop = img_rgb[faces[0][1]:faces[0][1]+faces[0][3],
                        faces[0][0]:faces[0][0]+faces[0][2]] if len(faces) > 0 else img_rgb

    resized   = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_float = resized.astype(np.float32)

    # Path A: simple [0,1] normalisation
    tensor_a = np.expand_dims(img_float / 255.0, axis=0)

    # Path B: ResNet50 channel-mean subtraction
    tensor_b = preprocess_input(np.expand_dims(img_float.copy(), axis=0))

    return tensor_a, tensor_b


# ---------------------------------------------------------------------------
# 3. CLINICAL SAFETY ADAPTER
# ---------------------------------------------------------------------------
_SAFE_STEPS = True   # set False to skip renumbering during unit tests

def _renumber(routine: list) -> list:
    """Re-assign step numbers sequentially after all inserts are done."""
    for i, item in enumerate(routine, start=1):
        item["step"] = i
    return routine


def assemble_safe_routine(predictions: np.ndarray, redness_score: float):
    """
    Maps the RF's 5-element binary prediction array to a clinically safe
    AM/PM/weekly regimen.

    Ingredient index contract (must match rf_targets order):
      0 → needs_salicylic_acid
      1 → needs_retinol
      2 → needs_vitamin_c
      3 → needs_niacinamide
      4 → needs_azelaic_acid

    Safety rules enforced:
      R1 — Salicylic + Retinol conflict: split to AM / PM
      R2 — Solo Retinol: start 2×/week, not daily (barrier protection)
      R3 — Niacinamide always AM (anti-inflammatory under SPF)
      R4 — Vitamin C always AM (antioxidant synergy with SPF)
      R5 — Azelaic Acid always PM (less photosensitising than Salicylic)
      R6 — Salicylic without Retinol: PM only (avoid daytime irritation)
      R7 — High redness (≥0.65): downgrade Salicylic to Azelaic if RF missed it
    """
    needs_sal = bool(predictions[0])
    needs_ret = bool(predictions[1])
    needs_vit = bool(predictions[2])
    needs_nia = bool(predictions[3])
    needs_aze = bool(predictions[4])

    # R7: safety override for very sensitive skin
    if needs_sal and redness_score >= 0.65:
        needs_sal = False
        needs_aze = True

    # Base routine (steps will be renumbered at the end)
    am_routine = [
        {"step": 0, "product": "Gentle Milk Cleanser",  "purpose": "Cleanse without stripping barrier"},
        {"step": 0, "product": "Mineral SPF 50+",        "purpose": "Broad-spectrum UV protection"},
    ]
    pm_routine = [
        {"step": 0, "product": "Oil Cleanser",           "purpose": "Remove SPF and makeup"},
        {"step": 0, "product": "Ceramide Moisturizer",   "purpose": "Restore skin barrier overnight"},
    ]
    weekly_treatments = []
    routine_class     = "BALANCED_MAINTENANCE"

    # ── Active ingredient slot assignment ──────────────────────────────────

    # R4: Vitamin C → AM treatment slot (insert before SPF)
    if needs_vit:
        am_routine.insert(1, {
            "step": 0,
            "product": "Vitamin C Serum (L-Ascorbic 15%)",
            "purpose": "Antioxidant protection, brightening dark spots and circles",
        })

    # R3: Niacinamide → AM (after Vitamin C if present)
    if needs_nia:
        am_routine.insert(2 if needs_vit else 1, {
            "step": 0,
            "product": "Niacinamide 10% Serum",
            "purpose": "Reduce inflammation, regulate sebum",
        })

    # R1: Salicylic + Retinol conflict — split across slots
    if needs_sal and needs_ret:
        routine_class = "ACNE_AND_AGING_REPAIR"
        am_routine.insert(1, {
            "step": 0,
            "product": "Salicylic Acid 2%",
            "purpose": "Exfoliate pores, control acne",
        })
        pm_routine.insert(1, {
            "step": 0,
            "product": "Encapsulated Retinol (0.2%)",
            "purpose": "Accelerate cell turnover, reduce fine lines",
        })

    # R6: Solo Salicylic → PM only
    elif needs_sal:
        routine_class = "ACNE_CONTROL"
        pm_routine.insert(1, {
            "step": 0,
            "product": "Salicylic Acid 2%",
            "purpose": "Exfoliate pores, control acne",
        })

    # R2: Solo Retinol → weekly ramp-up protocol, not daily PM
    elif needs_ret:
        routine_class = "ANTI_AGING_RENEWAL"
        weekly_treatments.append({
            "product": "Encapsulated Retinol (0.2%)",
            "frequency": "2× per week (increase to nightly after 4 weeks)",
            "slot": "PM — after moisturizer as the final step",
            "instructions": (
                "Apply a pea-sized amount. If stinging or peeling occurs, "
                "reduce to 1× per week and build up slowly."
            ),
        })

    # R5: Azelaic Acid → PM (gentler alternative for redness-prone skin)
    if needs_aze:
        if routine_class == "BALANCED_MAINTENANCE":
            routine_class = "ACNE_REDNESS_CONTROL"
        pm_routine.insert(1, {
            "step": 0,
            "product": "Azelaic Acid 15%",
            "purpose": "Reduce redness, control acne without irritation",
        })

    # ── Weekly treatments (condition-based boosters) ───────────────────────

    # Retinol solo already added above. Additional weekly boosters:

    # Acne patients benefit from a weekly clay mask to deep-clean pores
    if needs_sal or needs_aze:
        weekly_treatments.append({
            "product": "Kaolin Clay Mask",
            "frequency": "1× per week",
            "slot": "PM — after cleansing, before moisturizer",
            "instructions": (
                "Apply a thin layer to the face, leave for 10 minutes, "
                "rinse thoroughly with lukewarm water."
            ),
        })

    # Pigmentation patients benefit from a weekly chemical exfoliant
    if needs_vit:
        weekly_treatments.append({
            "product": "AHA 10% Exfoliating Toner (Glycolic Acid)",
            "frequency": "1–2× per week",
            "slot": "PM — after cleansing, before serums",
            "instructions": (
                "Apply with a cotton pad, do not rinse off. "
                "Do not use on the same night as Retinol. "
                "Always follow with SPF the next morning."
            ),
        })

    # Dark circles benefit from a weekly caffeine + peptide eye treatment
    if redness_score < 0.5 and not needs_aze:
        weekly_treatments.append({
            "product": "Caffeine + Peptide Eye Mask Patches",
            "frequency": "2× per week",
            "slot": "AM or PM — under-eye area only",
            "instructions": (
                "Apply patches to clean, dry under-eye skin. "
                "Leave for 15–20 minutes, pat in remaining serum."
            ),
        })

    # Everyone benefits from a weekly hydrating barrier mask
    weekly_treatments.append({
        "product": "Hyaluronic Acid Sheet Mask",
        "frequency": "1× per week",
        "slot": "PM — after cleansing, before moisturizer",
        "instructions": (
            "Apply to clean face for 15–20 minutes. "
            "Remove mask and pat remaining essence into skin. "
            "Follow with your regular PM moisturizer."
        ),
    })

    # Renumber all steps sequentially
    am_routine = _renumber(am_routine)
    pm_routine = _renumber(pm_routine)

    return routine_class, am_routine, pm_routine, weekly_treatments


# ---------------------------------------------------------------------------
# 4. CONFIDENCE SCORE  (from predict_proba, not hardcoded)
# ---------------------------------------------------------------------------
def compute_confidence(rf_model, rf_input: np.ndarray, predictions: np.ndarray) -> float:
    """
    Average probability of the predicted class across all active targets.
    Returns a value in [0.5, 1.0] — a hardcoded 0.92 is meaningless to C#.
    """
    try:
        # predict_proba returns a list of (n_samples, 2) arrays, one per target
        probas = rf_model.predict_proba(rf_input)   # list of 5 arrays
        confidences = []
        for i, pred in enumerate(predictions):
            # pred is 0 or 1; take the probability of the predicted class
            confidences.append(float(probas[i][0][int(pred)]))
        return round(float(np.mean(confidences)), 4)
    except Exception:
        return 0.0   # fallback if proba not available


# ---------------------------------------------------------------------------
# 5. MASTER PIPELINE
# ---------------------------------------------------------------------------
def analyze_face_pipeline(image_bytes: bytes, user_age: int = 25) -> dict:
    """
    Orchestrates Brain 1 → Brain 2 → Safety Adapter.
    Returns the JSON contract consumed by the C# / .NET 8.0 backend.
    """

    # ── Step 1: Image processing ───────────────────────────────────────────
    tensor_a, tensor_b = process_image(image_bytes)

    # ── Step 2: Brain 1 ONNX inference ────────────────────────────────────────
    if session_acne is None:
        raise RuntimeError("Brain 1 ONNX models are not loaded into memory.")

    # Execute ONNX sessions. 
    # NOTE: I am defaulting to tensor_b (ResNet50 preprocess). 
    # If any specific model was trained on the [0,1] normalized data, pass tensor_a instead.
    
    raw_acne         = session_acne.run(None, {in_acne: tensor_b})[0][0]
    raw_dark_spots   = session_dark_spots.run(None, {in_dark_spots: tensor_b})[0][0]
    raw_wrinkles     = session_wrinkles.run(None, {in_wrinkles: tensor_b})[0][0]
    raw_redness      = session_redness.run(None, {in_redness: tensor_b})[0][0]
    raw_dark_circles = session_dark_circle.run(None, {in_dark_circle: tensor_b})[0][0]
    raw_gender       = session_gender.run(None, {in_gender: tensor_b})[0][0]

    # Map raw float outputs (0.0 - 1.0) to 0-100 scores
    acne_val         = round(float(raw_acne[0] * 100), 2)
    dark_spots_val   = round(float(raw_dark_spots[0] * 100), 2)
    wrinkles_val     = round(float(raw_wrinkles[0] * 100), 2)
    redness_val      = round(float(raw_redness[0] * 100), 2)
    dark_circles_val = round(float(raw_dark_circles[0] * 100), 2)
    
    # Handle binary classification mapping for gender Female is 0 and Male is 1
    gender_num = 0 if raw_gender[0] < 0.5 else 1 
    gender_val = "Female" if gender_num == 0 else "Male"

    diagnostics = {
        "acne":         acne_val,
        "dark_spots":   dark_spots_val,
        "wrinkles":     wrinkles_val,
        "redness":      redness_val,
        "dark_circles": dark_circles_val,
        "gender":       gender_val,
    }
    
    # ── Step 3: Brain 2 — Random Forest inference ──────────────────────────
    routine_class  = "MANUAL_REVIEW"
    am, pm, weekly = [], [], []
    confidence     = 0.0

    if rf_model is not None:
        rf_input = pd.DataFrame([[
            user_age,
            gender_num,
            wrinkles_val     / 100.0,
            acne_val         / 100.0,
            redness_val      / 100.0,
            dark_circles_val / 100.0,
            dark_spots_val   / 100.0,
        ]], columns=rf_features)

        rf_predictions = rf_model.predict(rf_input)[0]
        confidence     = compute_confidence(rf_model, rf_input, rf_predictions)

        redness_norm = redness_val / 100.0
        routine_class, am, pm, weekly = assemble_safe_routine(
            rf_predictions, redness_norm
        )

    # ── Step 5: Build final JSON contract for C# ───────────────────────────
    return {
        "scanDate":     datetime.now().isoformat(),
        "routineClass": routine_class,
        "confidence":   confidence,
        "diagnostics":  diagnostics,
        "regimenSchedule": {
            "dailyAm":          am,
            "dailyPm":          pm,
            "weeklyTreatments": weekly,
        },
    }