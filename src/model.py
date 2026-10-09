"""10-convolution CAE with global residual learning, no pretrained weights."""
import argparse
import tensorflow as tf


DEFAULT_FILTERS = [32, 32, 64, 64, 96, 64, 64, 32, 32, 3]


def build_cae(filters=None):
    filters = list(DEFAULT_FILTERS if filters is None else filters)
    if len(filters) != 10 or filters[-1] != 3 or any(int(f) != f or f <= 0 for f in filters):
        raise ValueError("filters must contain 10 positive integers and end with RGB=3")
    inputs = tf.keras.Input((None, None, 3), name="noisy_rgb")
    x = inputs
    for i, channels in enumerate(filters, 1):
        if i in (6, 8):
            x = tf.keras.layers.UpSampling2D(2, interpolation="nearest", name=f"upsample_{i}")(x)
        x = tf.keras.layers.Conv2D(channels, 3, strides=2 if i in (2, 4) else 1,
                                  padding="same", activation="linear" if i == 10 else "relu",
                                  name="bottleneck" if i == 5 else f"conv_{i}")(x)
    # No clipping during optimization: raw RGB reconstruction = noisy + learned correction.
    output = tf.keras.layers.Add(name="residual_reconstruction")([inputs, x])
    return tf.keras.Model(inputs, output, name="residual_lightweight_cae")


def complexity(model, height=128, width=128):
    if height % 4 or width % 4:
        raise ValueError("Complexity input dimensions must be divisible by 4")
    h, w, channels = height, width, 3
    rows = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.UpSampling2D):
            h *= layer.size[0]
            w *= layer.size[1]
        elif isinstance(layer, tf.keras.layers.Conv2D):
            h = (h+layer.strides[0]-1)//layer.strides[0]
            w = (w+layer.strides[1]-1)//layer.strides[1]
            macs = h*w*layer.filters*layer.kernel_size[0]*layer.kernel_size[1]*channels
            rows.append({"layer": layer.name, "shape": [h,w,layer.filters], "macs": macs})
            channels = layer.filters
    macs = sum(r["macs"] for r in rows)
    return {"input_shape": [height,width,3], "parameters": model.count_params(),
            "conv_macs": macs, "conv_flops": 2*macs, "layers": rows,
            "convention": "1 multiply-accumulate = 1 MAC = 2 FLOPs; convolution only; excludes bias, ReLU, upsampling, residual add and clipping"}


def print_summary(model):
    model.summary()
    info = complexity(model)
    print(f"128x128 RGB: {info['conv_macs']:,} MACs; {info['conv_flops']:,} FLOPs")
    print(info["convention"])
    return info


def main():
    from .common import load_config
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    args = p.parse_args()
    print_summary(build_cae(load_config(args.config)["model"]["filters"]))


if __name__ == "__main__":
    from .common import run_cli
    run_cli(main)
