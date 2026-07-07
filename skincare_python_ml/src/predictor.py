import cv2
import numpy as np
import pandas as pd
import joblib
import onnx
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
    model_path = r"D:\code\Capstone_project\skincare_python_ml\brain2_random_forest.pkl"
    brain2_artifact = joblib.load(model_path)

    rf_model    = brain2_artifact['model']
    rf_features = brain2_artifact['features']   # exact feature order
    rf_targets  = brain2_artifact['targets']    # exact target order
    print("✅ Brain 2 loaded successfully.")
except Exception as e:
    print(f"❌ Brain 2 load failed: {e}")
    rf_model = rf_features = rf_targets = None

# Helper to inject intermediate layers into ONNX graph in-memory on startup
def load_onnx_session_with_features(relative_path, providers):
    """
    Loads an ONNX model from disk, appends its last Conv output node in-memory, 
    and returns both the active session and the target conv tensor name.
    """
    onnx_base = r"D:\code\Capstone_project\skincare_python_ml"
    full_path = onnx_base + relative_path
    model = onnx.load(full_path)
    
    # Locate the final convolutional node automatically
    conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
    if not conv_nodes:
        raise ValueError(f"No Conv nodes found inside model graph at: {relative_path}")
        
    last_conv_tensor = conv_nodes[-1].output[0]
    
    # Inject intermediate tracking layer into the output graph definitions
    new_output = onnx.helper.make_tensor_value_info(last_conv_tensor, onnx.TensorProto.FLOAT, None)
    model.graph.output.append(new_output)
    
    # Compile directly to session string memory without writing a file to disk
    session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    return session, last_conv_tensor

# Load Brain 1 ONNX models with Heatmap feature trackers attached
try:
    providers = ['CPUExecutionProvider']
    onnx_base = r"D:\code\Capstone_project\skincare_python_ml"
 
    session_acne, acne_conv         = load_onnx_session_with_features(r"\acne_keras\acne_mvp_model.onnx", providers)
    session_dark_spots, spots_conv   = load_onnx_session_with_features(r"\dark_spots_keras\dark_spots_phase1.onnx", providers)
    session_wrinkles, wrinkles_conv = load_onnx_session_with_features(r"\wrinkles_keras\wrinkles_v2_production.onnx", providers)
    session_redness, redness_conv   = load_onnx_session_with_features(r"\redness_keras\redness_v1_production.onnx", providers)
    session_dark_circle, circles_conv = load_onnx_session_with_features(r"\dark_circle_keras\dark_circle_final.onnx", providers)
    
    # Gender model stays standard (no heatmaps needed)
    session_gender                  = ort.InferenceSession(onnx_base + r"\gender_keras\gender_production_fixed.onnx", providers=providers)
 
    # Cache input names for prediction mapping
    in_acne        = session_acne.get_inputs()[0].name
    in_dark_spots  = session_dark_spots.get_inputs()[0].name
    in_wrinkles    = session_wrinkles.get_inputs()[0].name
    in_redness     = session_redness.get_inputs()[0].name
    in_dark_circle = session_dark_circle.get_inputs()[0].name
    in_gender      = session_gender.get_inputs()[0].name
 
    print("✅ Brain 1 (ONNX) loaded successfully with intermediate feature maps.")
except Exception as e:
    print(f"❌ Brain 1 ONNX load failed: {e}")
    session_acne = session_dark_spots = session_wrinkles = None
    session_redness = session_dark_circle = session_gender = None

# ---------------------------------------------------------------------------
# 2. IMAGE PROCESSING (Streamlined & Cleaned)
# ---------------------------------------------------------------------------
def process_image(image_bytes: bytes):
    """
    Decodes image bytes and resizes directly to 224x224.
    Face alignment validation is pre-handled by MediaPipe Tasks in main.py.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure the upload is a valid JPEG/PNG.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Directly resize the valid face frame to your target dimensions
    resized   = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_float = resized.astype(np.float32)

    # Path A: simple [0,1] normalisation
    tensor_a = np.expand_dims(img_float / 255.0, axis=0)

    # Path B: ResNet50 channel-mean subtraction
    tensor_b = preprocess_input(np.expand_dims(img_float.copy(), axis=0))

    return tensor_a, tensor_b

# ---------------------------------------------------------------------------
# 3. FIXED HIGH-PERFORMANCE HEATMAP EXTRACTOR PIPELINE UTILITY
# ---------------------------------------------------------------------------
def extract_pipeline_heatmap(session, last_conv_tensor, target_node_name, model_input):
    """
    Computes a forward pass on an open session, isolates convolutional tensors safely,
    applies adaptive 40% noise reduction, and outputs a 50,176 element 1D array.
    """
    try:
        # 1. Gather all output names from the current session runtime
        output_names = [o.name for o in session.get_outputs()]
        
        # Verify both target tensors exist inside the session graph mapping
        if last_conv_tensor not in output_names:
            raise ValueError(f"Target node '{last_conv_tensor}' is missing from the session outputs.")
            
        pred_node = [n for n in output_names if n != last_conv_tensor][0]
        
        # Run inference and map output names directly to separate variables
        outputs = session.run([pred_node, last_conv_tensor], {target_node_name: model_input})
        
        # Create a clean dictionary mapping tensor names to their raw array results
        output_map = dict(zip([pred_node, last_conv_tensor], outputs))
        
        # FIXED: Extract ONLY the dedicated convolutional tensor array explicitly by name!
        f_map = output_map[last_conv_tensor][0]
        
        # 2. Dynamically manage channels layout format matching (NCHW vs NHWC)
        if f_map.shape[0] == 2048 or f_map.shape[0] == 512:
            f_map = np.transpose(f_map, (1, 2, 0))
            
        # Max-pooling isolates focal feature changes while filtering background room haze
        heatmap_raw = np.max(f_map, axis=-1)
        heatmap_raw = np.maximum(heatmap_raw, 0) # Apply clean structural ReLU
        
        # 3. Bounding scale normalization [0, 1]
        max_val = np.max(heatmap_raw)
        min_val = np.min(heatmap_raw)
        if max_val - min_val > 1e-8:
            heatmap_raw = (heatmap_raw - min_val) / (max_val - min_val)
            
        # Apply your verified 40% threshold filter cutoff to erase outer environment borders
        heatmap_raw[heatmap_raw < 0.40] = 0.0
        
        # 4. Upsample directly to standard data grid bounds canvas (224, 224)
        heatmap_224 = cv2.resize(heatmap_raw, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Stretch active regions evenly across space bounds 
        final_max = np.max(heatmap_224)
        if final_max > 0:
            heatmap_224 = heatmap_224 / final_max
            
        # Flatten array matrix grid into 1D data list contract for database transmission
        return heatmap_224.flatten().round(4).tolist()

    except Exception as e:
        print(f"⚠️ Heatmap production failure on node {last_conv_tensor}: {e}")
        return np.zeros(50176, dtype=np.float32).tolist()

# ---------------------------------------------------------------------------
# 4. CLINICAL SAFETY ADAPTER
# ---------------------------------------------------------------------------
_SAFE_STEPS = True

def _renumber(routine: list) -> list:
    for i, item in enumerate(routine, start=1):
        item["step"] = i
    return routine

def assemble_safe_routine(predictions: np.ndarray, redness_score: float):
    needs_sal = bool(predictions[0])
    needs_ret = bool(predictions[1])
    needs_vit = bool(predictions[2])
    needs_nia = bool(predictions[3])
    needs_aze = bool(predictions[4])

    if needs_sal and redness_score >= 0.65:
        needs_sal = False
        needs_aze = True

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

    if needs_vit:
        am_routine.insert(1, {
            "step": 0,
            "product": "Vitamin C Serum (L-Ascorbic 15%)",
            "purpose": "Antioxidant protection, brightening dark spots and circles",
        })

    if needs_nia:
        am_routine.insert(2 if needs_vit else 1, {
            "step": 0,
            "product": "Niacinamide 10% Serum",
            "purpose": "Reduce inflammation, regulate sebum",
        })

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
    elif needs_sal:
        routine_class = "ACNE_CONTROL"
        pm_routine.insert(1, {
            "step": 0,
            "product": "Salicylic Acid 2%",
            "purpose": "Exfoliate pores, control acne",
        })
    elif needs_ret:
        routine_class = "ANTI_AGING_RENEWAL"
        weekly_treatments.append({
            "product": "Encapsulated Retinol (0.2%)",
            "frequency": "2× per week (increase to nightly after 4 weeks)",
            "slot": "PM — after moisturizer as the final step",
            "instructions": "Apply a pea-sized amount. If stinging occurs, reduce frequency.",
        })

    if needs_aze:
        if routine_class == "BALANCED_MAINTENANCE":
            routine_class = "ACNE_REDNESS_CONTROL"
        pm_routine.insert(1, {
            "step": 0,
            "product": "Azelaic Acid 15%",
            "purpose": "Reduce redness, control acne without irritation",
        })

    if needs_sal or needs_aze:
        weekly_treatments.append({
            "product": "Kaolin Clay Mask",
            "frequency": "1× per week",
            "slot": "PM — after cleansing, before moisturizer",
            "instructions": "Apply a thin layer to the face, leave for 10 minutes, rinse thoroughly.",
        })

    if needs_vit:
        weekly_treatments.append({
            "product": "AHA 10% Exfoliating Toner (Glycolic Acid)",
            "frequency": "1–2× per week",
            "slot": "PM — after cleansing, before serums",
            "instructions": "Apply with a cotton pad, do not rinse off. Do not use on same night as Retinol.",
        })

    if redness_score < 0.5 and not needs_aze:
        weekly_treatments.append({
            "product": "Caffeine + Peptide Eye Mask Patches",
            "frequency": "2× per week",
            "slot": "AM or PM — under-eye area only",
            "instructions": "Apply patches to clean, dry under-eye skin. Leave for 15–20 minutes.",
        })

    weekly_treatments.append({
        "product": "Hyaluronic Acid Sheet Mask",
        "frequency": "1× per week",
        "slot": "PM — after cleansing, before moisturizer",
        "instructions": "Apply to clean face for 15–20 minutes, then follow with regular PM moisturizer.",
    })

    am_routine = _renumber(am_routine)
    pm_routine = _renumber(pm_routine)

    return routine_class, am_routine, pm_routine, weekly_treatments

# ---------------------------------------------------------------------------
# 5. CONFIDENCE SCORE
# ---------------------------------------------------------------------------
def compute_confidence(rf_model, rf_input: np.ndarray, predictions: np.ndarray) -> float:
    """
    Average probability of the predicted class across all active targets.
    Returns a value in [0.5, 1.0] — a hardcoded 0.92 is meaningless to C#.
    """
    try:
        # predict_proba returns a list of (n_samples, 2) arrays, one per target
        probas = rf_model.predict_proba(rf_input)
        confidences = []
        for i, pred in enumerate(predictions):
            # pred is 0 or 1; take the probability of the predicted class
            confidences.append(float(probas[i][0][int(pred)]))
        return round(float(np.mean(confidences)), 4)
    except Exception:
        return 0.0  # fallback if proba not available
    
# ---------------------------------------------------------------------------
# 6. MASTER PIPELINE
# ---------------------------------------------------------------------------
def analyze_face_pipeline(image_bytes: bytes, user_age) -> dict:
    """
    Orchestrates Brain 1 -> Spatial Heatmap Computations -> Brain 2 -> JSON Contract.
    """
    # ── Step 1: Image processing ───────────────────────────────────────────
    tensor_a, tensor_b = process_image(image_bytes)

    if session_acne is None:
        raise RuntimeError("Brain 1 ONNX models are not loaded into memory.")

    # Cast to strict float32 precision arrays to prevent double validation typing crashes
    onnx_input_tensor = tensor_b.astype(np.float32)

    # ── Step 2: Spatial Feature Heatmap Generation ─────────────────────────
    # Run the optimized channel-aggregation extraction logic across all 5 models simultaneously
    print("[Inference] Computing diagnostic heatmaps via raw forward passes...")
    acne_map        = extract_pipeline_heatmap(session_acne, acne_conv, in_acne, onnx_input_tensor)
    dark_spots_map  = extract_pipeline_heatmap(session_dark_spots, spots_conv, in_dark_spots, onnx_input_tensor)
    wrinkles_map    = extract_pipeline_heatmap(session_wrinkles, wrinkles_conv, in_wrinkles, onnx_input_tensor)
    redness_map     = extract_pipeline_heatmap(session_redness, redness_conv, in_redness, onnx_input_tensor)
    dark_circles_map = extract_pipeline_heatmap(session_dark_circle, circles_conv, in_dark_circle, onnx_input_tensor)

    # ── Step 3: Diagnostic Scoring Calculations ────────────────────────────
    # Extract baseline prediction nodes to compute severity percentages
    raw_acne         = session_acne.run(None, {in_acne: onnx_input_tensor})[0][0]
    raw_dark_spots   = session_dark_spots.run(None, {in_dark_spots: onnx_input_tensor})[0][0]
    raw_wrinkles     = session_wrinkles.run(None, {in_wrinkles: onnx_input_tensor})[0][0]
    raw_redness      = session_redness.run(None, {in_redness: onnx_input_tensor})[0][0]
    raw_dark_circles = session_dark_circle.run(None, {in_dark_circle: onnx_input_tensor})[0][0]
    raw_gender       = session_gender.run(None, {in_gender: onnx_input_tensor})[0][0]

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
    
    # ── Step 4: Brain 2 — Random Forest recommendation engine ──────────────
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

    # ── Step 5: Build extended data contract for ASP.NET ───────────────────
    return {
        "scanDate":     datetime.now().isoformat(),
        "routineClass": routine_class,
        "confidence":   confidence,
        "diagnostics":  diagnostics,
        "heatmaps": {
            "acne":        acne_map,
            "darkSpots":   dark_spots_map,
            "wrinkles":    wrinkles_map,
            "redness":     redness_map,
            "darkCircles": dark_circles_map
        },
        "regimenSchedule": {
            "dailyAm":          am,
            "dailyPm":          pm,
            "weeklyTreatments": weekly,
        },
    }