import numpy as np
import cv2
import onnx
import onnxruntime as ort

def run_independent_test():
    print("=" * 60)
    print("🚀 RUNNING STANDALONE CHANNEL-AGGREGATION HEATMAP TEST")
    print("=" * 60)

    # 1. Define your exact local file paths
    # UPDATE THESE LINES TO MATCH YOUR REAL FOLDER NAMING LAYOUT
    model_path = r"D:\code\Capstone_project\skincare_python_ml\acne_keras\acne_mvp_model.onnx"
    image_path = r"D:\code\Capstone_project\skincare_python_ml\test.jpg"
    output_path = r"D:\code\Capstone_project\skincare_python_ml\src\optimized_acne_heatmap.jpg"

    try:
        # 2. Load the ONNX graph and append the last conv layer as an extra output
        print("[1/5] Loading ONNX model graph...")
        model = onnx.load(model_path)
        
        conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
        if not conv_nodes:
            raise ValueError("No Convolutional nodes found in this ONNX model graph.")
            
        last_conv_tensor = conv_nodes[-1].output[0]
        print(f"      Targeting last Conv output tensor: '{last_conv_tensor}'")

        new_output = onnx.helper.make_tensor_value_info(last_conv_tensor, onnx.TensorProto.FLOAT, None)
        model.graph.output.append(new_output)

        # 3. Create inference session
        print("[2/5] Initializing ONNX Runtime Session...")
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        pred_node = [n for n in output_names if n != last_conv_tensor][0]

        # 4. Load and preprocess the real test image
        print("[3/5] Loading and preprocessing 'test.jpg'...")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image file at: {image_path}")
            
        img_resized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        # ResNet50 exact channel-mean subtraction matching your original framework
        mean = np.array([103.939, 116.779, 123.68]) # BGR ImageNet constants
        model_input = img_resized - mean
        
        # FIX: Explicitly cast to float32 to prevent automatic NumPy conversion to double precision
        model_input = np.expand_dims(model_input, axis=0).astype(np.float32)

        # 5. Execute ONE single forward pass
        print("[4/5] Executing optimized single forward inference pass...")
        pred, feature_maps = session.run([pred_node, last_conv_tensor], {input_name: model_input})
        print(f"      Prediction classification shape: {pred.shape}")
        print(f"      Raw feature map matrix shape: {feature_maps.shape}")

        # 6. Channel Aggregation Processing (FIXED FOR ACTIVE FEATURES)
        f_map = feature_maps[0]
        if f_map.shape[0] == 2048:  # If layout is NCHW (Channels-First), transpose to NHWC
            f_map = np.transpose(f_map, (1, 2, 0))

        # CRITICAL CHANGE: Use max pooling across channels instead of mean pooling
        # This grabs ONLY the strongest feature activations and ignores background noise
        heatmap_raw = np.max(f_map, axis=-1)
        
        # Apply ReLU filter to eliminate negative activations
        heatmap_raw = np.maximum(heatmap_raw, 0)

        # Bounds scale normalization [0, 1]
        max_val = np.max(heatmap_raw)
        min_val = np.min(heatmap_raw)
        if max_val - min_val > 1e-8:
            heatmap_raw = (heatmap_raw - min_val) / (max_val - min_val)

        # Smooth upscale back up to original image dimensions
        heatmap_224 = cv2.resize(heatmap_raw, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        # ===================================================================
        # 7. GENERATE A SPECIFIC MONOCHROME COLOR MASK (For Independent Visual Check)
        # ===================================================================
        # Create an empty black image matching your original image dimensions
        blank_mask = np.zeros_like(img_bgr)

        # CHOOSE YOUR IDENTITY COLOR HERE (BGR Format for OpenCV):
        # Acne (Pure Red):       [0, 0, 255]
        # Wrinkles (Pure Purple): [255, 0, 128]
        # Redness (Pure Green):   [0, 255, 0]
        identity_color = np.array([0, 0, 255]) # Let's test Acne Red

        # Paint the intensity values smoothly with your target color choice
        # heatmap_224 contains values from 0.0 to 1.0 acting as our alpha channel
        for i in range(3): # Loop through B, G, R channels
            blank_mask[:, :, i] = np.uint8(heatmap_224 * identity_color[i])

        # Blend the raw image with your new single-color monochrome overlay mask
        # 0.7 keeps the face clear, 0.5 controls how intensely the color glows
        overlaid_output = cv2.addWeighted(img_bgr, 0.7, blank_mask, 0.5, 0)

        # Save to disk
        cv2.imwrite(output_path, overlaid_output)
        print(f"[done] Monochromatic test overlay saved to: {output_path}")

        # Blend the raw image with the glowing heatmap overlay
        overlaid_output = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

        # Save to disk
        cv2.imwrite(output_path, overlaid_output)
        print("[5/5] Success! Heatmap compiled cleanly without artifacts.")
        print(f"👉 Saved output layout to: {output_path}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Test Failed: {str(e)}")
        print("=" * 60)

if __name__ == "__main__":
    run_independent_test()
