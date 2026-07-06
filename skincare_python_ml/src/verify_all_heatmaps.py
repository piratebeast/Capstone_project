import numpy as np
import cv2
import onnx
import onnxruntime as ort
import os

def generate_targeted_heatmap(model_path, image_path, color_bgr):
    """Loads a model, extracts its spatial layers using max-pooling to catch strong features,

    cleans background wall noise, and applies a single monochrome color layer.
    """
    # 1. In-memory graph modification
    model = onnx.load(model_path)
    conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
    last_conv_tensor = conv_nodes[-1].output[0]
    model.graph.output.append(onnx.helper.make_tensor_value_info(last_conv_tensor, onnx.TensorProto.FLOAT, None))
    
    # 2. Run forward pass
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    pred_node = [n for n in output_names if n != last_conv_tensor][0]
    
    # Preprocess image
    img_bgr = cv2.imread(image_path)
    h_orig, w_orig, _ = img_bgr.shape
    img_resized = cv2.resize(img_bgr, (224, 224)).astype(np.float32)
    mean = np.array([103.939, 116.779, 123.68]) # ResNet50 defaults
    model_input = np.expand_dims(img_resized - mean, axis=0).astype(np.float32)
    
    _, feature_maps = session.run([pred_node, last_conv_tensor], {input_name: model_input})
    
    # 3. Process Spatial Maps (Dynamic Reshaping)
    f_map = feature_maps[0]
    if f_map.shape[0] == 2048 or f_map.shape[0] == 512: # NCHW check
        f_map = np.transpose(f_map, (1, 2, 0))
        
    # FIX: Restored max pooling across channels to keep the strong feature spikes
    heatmap_raw = np.max(f_map, axis=-1)
    heatmap_raw = np.maximum(heatmap_raw, 0) # ReLU
    
    # Normalize bounds [0, 1] before noise filtration
    max_val = np.max(heatmap_raw)
    min_val = np.min(heatmap_raw)
    if max_val - min_val > 1e-8:
        heatmap_raw = (heatmap_raw - min_val) / (max_val - min_val)
        
    # Clean background room noise while allowing actual facial hits to stay active
    # This zeroes out any weak activations under 40% intensity globally
    heatmap_raw[heatmap_raw < 0.40] = 0.0
    
    # Upscale smoothly to full image size
    heatmap_scaled = cv2.resize(heatmap_raw, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    # 4. Generate Single Monochrome Color Mask
    blank_mask = np.zeros_like(img_bgr)
    for i in range(3): # Apply targeted BGR profile channel-by-channel
        blank_mask[:, :, i] = np.uint8(heatmap_scaled * color_bgr[i])
        
    # Blend with original image (0.75 face transparency, 0.45 glow intensity)
    return cv2.addWeighted(img_bgr, 0.75, blank_mask, 0.45, 0)

def main():
    onnx_base = r"D:\code\Capstone_project\skincare_python_ml"
    image_path = r"D:\code\Capstone_project\skincare_python_ml\test3.jpg"
    output_dir = r"D:\code\Capstone_project\skincare_python_ml\src\test_outputs3"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define distinct clinical colors for each skin issue (BGR Format for OpenCV)
    models_config = {
        "acne": {
            "path": r"\acne_keras\acne_mvp_model.onnx",
            "color": [0, 0, 255] # 🔴 Red
        },
        "wrinkles": {
            "path": r"\wrinkles_keras\wrinkles_v2_production.onnx",
            "color": [255, 0, 128] # 🟣 Purple
        },
        "redness": {
            "path": r"\redness_keras\redness_v1_production.onnx",
            "color": [0, 255, 0] # 🟢 Green
        },
        "dark_spots": {
            "path": r"\dark_spots_keras\dark_spots_phase1.onnx",
            "color": [0, 140, 255] # 🟠 Orange
        },
        "dark_circles": {
            "path": r"\dark_circle_keras\dark_circle_final.onnx",
            "color": [255, 255, 0] # 🔵 Cyan / Blue
        }
    }
    
    print("=" * 70)
    print("🎨 GENERATING CORRECTED MULTI-MODEL MONOCHROME SUITE")
    print("=" * 70)
    
    for name, config in models_config.items():
        full_model_path = onnx_base + config["path"]
        out_file = os.path.join(output_dir, f"focused_{name}_heatmap.jpg")
        print(f"🔄 Processing clean colored overlay for: [{name.upper()}]...")
        
        try:
            result_img = generate_targeted_heatmap(full_model_path, image_path, config["color"])
            cv2.imwrite(out_file, result_img)
            print(f"   ✅ Saved -> {out_file}")
        except Exception as e:
            print(f"   ❌ Error processing {name}: {str(e)}")
            
    print("\n🏁 All tests completed. Check your 'src/test_outputs' folder!")

if __name__ == "__main__":
    main()