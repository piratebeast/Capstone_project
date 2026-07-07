"""
Score-CAM heatmap generator that works purely through ONNX Runtime
forward passes -- no PyTorch, no gradients, no onnx2torch.

Stays inside your existing ONNX-only stack, so it works identically
for all 5 models regardless of whether they came from Keras 2 or Keras 3.

Two stages:
  1. Inspect / modify the ONNX graph to expose the last conv layer's
     output as an extra model output (done in-memory, nothing saved
     to disk unless you ask for it).
  2. Run Score-CAM: extract feature maps with one forward pass, mask
     the input image with each (upsampled) feature map, batch-run
     forward passes to get a class score per channel, then combine.

Install once:
    pip install onnx onnxruntime opencv-python numpy

Usage:
    python scorecam_onnx.py --model acne_model.onnx --image test.jpg --out acne_heatmap.jpg
"""

import argparse
import numpy as np
import cv2
import onnx
import onnxruntime as ort


def find_last_conv_node(onnx_model):
    """
    Scan the ONNX graph and return the output tensor name of the last
    Conv node. This is what we'll tap as our 'feature map' layer for
    Score-CAM -- the ONNX equivalent of Grad-CAM's last conv layer.
    """
    conv_nodes = [n for n in onnx_model.graph.node if n.op_type == "Conv"]
    if not conv_nodes:
        raise ValueError("No Conv nodes found in this ONNX graph. "
                          "Open the model in Netron to find the right "
                          "internal tensor name manually.")
    last_conv = conv_nodes[-1]
    output_name = last_conv.output[0]
    print(f"[info] Last Conv node: '{last_conv.name}' -> output tensor '{output_name}'")
    return output_name


def build_session_with_feature_output(onnx_path, feature_tensor_name):
    """
    Load the ONNX model, add the chosen intermediate tensor as an extra
    graph output (in-memory only), and return an InferenceSession that
    now returns BOTH the original prediction AND the feature maps.
    """
    model = onnx.load(onnx_path)
    feature_output_name = find_last_conv_node(model) if feature_tensor_name is None else feature_tensor_name

    # Add the intermediate tensor as a new graph output (shape left
    # unspecified -- ONNX Runtime will infer it at run time).
    new_output = onnx.helper.make_tensor_value_info(
        feature_output_name, onnx.TensorProto.FLOAT, None
    )
    model.graph.output.append(new_output)

    session = ort.InferenceSession(model.SerializeToString(),
                                    providers=["CPUExecutionProvider"])
    return session, feature_output_name


def preprocess_image(img_path, target_size=(224, 224)):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original = img_rgb.copy()

    img_resized = cv2.resize(img_rgb, target_size).astype(np.float32)
    img_for_overlay = img_resized / 255.0

    # NOTE: match this to whatever preprocess_input your Keras models used.
    # This is plain ImageNet mean-centering as a starting point.
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    model_input = img_resized - mean  # NHWC, matches typical tf2onnx export
    model_input = np.expand_dims(model_input, axis=0).astype(np.float32)

    return model_input, original, img_for_overlay


def run_scorecam(model_path, image_path, out_path, target_size=(224, 224),
                  class_index=None, batch_size=32, feature_tensor_name=None):

    session, feature_name = build_session_with_feature_output(model_path, feature_tensor_name)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    pred_output_name = [n for n in output_names if n != feature_name][0]

    model_input, original_img, img_overlay = preprocess_image(image_path, target_size)

    # Stage 1: single forward pass to get feature maps + base prediction
    pred, feature_maps = session.run([pred_output_name, feature_name], {input_name: model_input})
    print(f"[info] Prediction output: {pred}")
    print(f"[info] Feature map shape: {feature_maps.shape}  (NHWC assumed: batch,H,W,C)")

    if class_index is None:
        class_index = 0 if pred.shape[-1] == 1 else int(np.argmax(pred[0]))

    # Assume NHWC layout (typical for tf2onnx-exported Keras models).
    # If your feature_maps.shape looks like (1, C, H, W) instead, flip
    # this to feature_maps[0].transpose(1, 2, 0)
    fmap = feature_maps[0]  # (H, W, C)
    h, w, num_channels = fmap.shape

    scores = np.zeros(num_channels, dtype=np.float32)

    batch_inputs = []
    batch_indices = []

    def flush_batch():
        if not batch_inputs:
            return
        batch = np.concatenate(batch_inputs, axis=0)
        preds = session.run([pred_output_name], {input_name: batch})[0]
        for idx, p in zip(batch_indices, preds):
            scores[idx] = p[class_index] if p.shape[-1] > 1 else p[0]
        batch_inputs.clear()
        batch_indices.clear()

    for c in range(num_channels):
        channel_map = fmap[:, :, c]
        channel_map = cv2.resize(channel_map, target_size)
        # normalize to [0,1]
        cmin, cmax = channel_map.min(), channel_map.max()
        if cmax - cmin < 1e-8:
            continue
        norm_map = (channel_map - cmin) / (cmax - cmin)

        masked = model_input[0] * norm_map[..., np.newaxis]
        batch_inputs.append(np.expand_dims(masked, axis=0))
        batch_indices.append(c)

        if len(batch_inputs) == batch_size:
            flush_batch()
    flush_batch()

    # Softmax-normalize channel scores as weights
    weights = scores - np.max(scores)
    weights = np.exp(weights)
    weights = weights / (np.sum(weights) + 1e-8)

    heatmap = np.zeros((h, w), dtype=np.float32)
    for c in range(num_channels):
        heatmap += weights[c] * fmap[:, :, c]

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    heatmap_resized = cv2.resize(heatmap, target_size)

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (img_overlay * 255).astype(np.uint8)
    overlaid = cv2.addWeighted(overlay, 0.6, heatmap_color, 0.4, 0)
    overlaid_resized = cv2.resize(overlaid, (original_img.shape[1], original_img.shape[0]))

    overlaid_bgr = cv2.cvtColor(overlaid_resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, overlaid_bgr)
    print(f"[done] Heatmap saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .onnx model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", required=True, help="Path to save output heatmap image")
    parser.add_argument("--class_index", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_tensor_name", default=None,
                         help="Manually override the last-conv tensor name if auto-detect picks the wrong node")
    args = parser.parse_args()

    run_scorecam(args.model, args.image, args.out,
                 class_index=args.class_index,
                 batch_size=args.batch_size,
                 feature_tensor_name=args.feature_tensor_name)
    
    