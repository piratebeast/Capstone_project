import numpy as np
import cv2
import onnx
import onnxruntime as ort

def test_entire_suite():
    print("=" * 70)
    print("🔍 RUNNING MULTI-MODEL INTERMEDIATE LAYER HEALTH CHECK")
    print("=" * 70)

    # 1. Map out all 5 model locations matching your folder structure
    onnx_base = r"D:\code\Capstone_project\skincare_python_ml"
    models_to_check = {
        "Acne": r"\acne_keras\acne_mvp_model.onnx",
        "Dark Spots": r"\dark_spots_keras\dark_spots_phase1.onnx",
        "Wrinkles": r"\wrinkles_keras\wrinkles_v2_production.onnx",
        "Redness": r"\redness_keras\redness_v1_production.onnx",
        "Dark Circles": r"\dark_circle_keras\dark_circle_final.onnx"
    }

    image_path = r"D:\code\Capstone_project\skincare_python_ml\test.jpg"
    
    # Preprocess a single dummy model input matching float32 layout specifications
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Error: Cannot find test image at {image_path}")
        return
        
    img_resized = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    mean = np.array([103.939, 116.779, 123.68])
    model_input = np.expand_dims(img_resized - mean, axis=0).astype(np.float32)

    # 2. Loop through and validate every network graph pipeline execution
    for model_name, relative_path in models_to_check.items():
        full_path = onnx_base + relative_path
        print(f"\n🔄 Testing [{model_name}] Graph Model...")
        
        try:
            # Load graph string structure
            model = onnx.load(full_path)
            
            # Extract last convolutional node output identity
            conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
            if not conv_nodes:
                print(f"   ⚠️ Warning: No Conv node found inside {model_name}. Skipping.")
                continue
                
            last_conv_tensor = conv_nodes[-1].output[0]
            
            # Inject custom intermediate channel output tracking info in-memory
            new_output = onnx.helper.make_tensor_value_info(last_conv_tensor, onnx.TensorProto.FLOAT, None)
            model.graph.output.append(new_output)
            
            # Start inference worker instance block
            session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            output_names = [o.name for o in session.get_outputs()]
            pred_node = [n for n in output_names if n != last_conv_tensor][0]
            
            # Trigger single forward check
            _, feature_maps = session.run([pred_node, last_conv_tensor], {input_name: model_input})
            
            f_map = feature_maps[0]
            if f_map.shape[0] == 2048 or f_map.shape[0] == 512: # Handle varying channels count
                f_map = np.transpose(f_map, (1, 2, 0))
                
            heatmap_raw = np.max(f_map, axis=-1)
            print(f"   ✅ SUCCESS: Feature map shape is {feature_maps.shape}. Processing is stable.")
            
        except Exception as e:
            print(f"   ❌ FAILED: Graph structure mismatch error -> {str(e)}")

    print("\n" + "=" * 70)
    print("🏁 DIAGNOSTIC COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    test_entire_suite()