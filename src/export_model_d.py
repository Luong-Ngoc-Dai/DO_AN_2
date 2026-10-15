"""Export Model D FP32 kernels/biases for a future verified quantization/RTL flow.

This DOES NOT quantize weights or create a directly synthesizable FPGA bitstream.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from .model_d import build_paper_guided_cae
from .common import run_cli


def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def export_model_d(model_path, output):
    path = Path(model_path)
    model = tf.keras.models.load_model(path, compile=False)
    convs = [x for x in model.layers if isinstance(x,tf.keras.layers.Conv2D)]
    # Counts alone also accept altered activations, strides and skip wiring.
    # Validate against the supplied graph, without embedding expected costs here.
    canonical = lambda graph: json.loads(json.dumps(graph.get_config()))
    if canonical(model) != canonical(build_paper_guided_cae()):
        raise ValueError("Checkpoint is not the expected Model D graph")
    tensors = []
    for layer in convs:
        arrays = layer.get_weights()
        if any(w.dtype != np.float32 or not np.all(np.isfinite(w)) for w in arrays):
            raise ValueError(f"{layer.name}: expected finite FP32 weights and biases")
        tensors.append(arrays)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"model":"paper_guided_7conv_cae", "checkpoint_sha256":sha256(path),
                "input_layout":"NHWC", "kernel_layout":"HWIO", "flatten_order":"C",
                "dtype":"float32", "quantized":False,
                "note":"TensorFlow Conv2D is cross-correlation; do not flip kernel axes."
                       " Fixed-point scale/rounding/saturation must be defined before FPGA inference.",
                "layers":[]}
    for i,(layer,arrays) in enumerate(zip(convs,tensors),1):
        kernel,bias = arrays
        base = f"conv_{i:02d}"
        np.save(output/f"{base}_kernel.npy",kernel)
        np.save(output/f"{base}_bias.npy",bias)
        manifest["layers"].append({"index":i,"name":layer.name,
                                   "kernel_shape":list(kernel.shape),"bias_shape":list(bias.shape),
                                   "kernel_npy":f"{base}_kernel.npy","bias_npy":f"{base}_bias.npy",
                                   "kernel_sha256":sha256(output/f"{base}_kernel.npy"),
                                   "bias_sha256":sha256(output/f"{base}_bias.npy")})
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    print("Saved FP32 tensors:",export_model_d(args.model,args.output))

if __name__ == "__main__":
    run_cli(main)
