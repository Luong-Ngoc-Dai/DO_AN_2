"""10-convolution CAE with global residual learning, no pretrained weights."""
import argparse
import tensorflow as tf


DEFAULT_FILTERS = [
    32, 32, 64, 64,  # Encoder: Conv 1–4.
    96,              # Bottleneck: Conv 5.
    64, 64, 32, 32, 3,  # Decoder: Conv 6–10.
]


def build_cae(filters=None):
    """Khai báo 10 Conv2D tường minh; filters có thể đọc từ baseline.yaml."""
    filters = list(DEFAULT_FILTERS if filters is None else filters)
    if len(filters) != 10 or filters[-1] != 3 or any(int(f) != f or f <= 0 for f in filters):
        raise ValueError("filters must contain 10 positive integers and end with RGB=3")
    # Input RGB; chiều cao/rộng linh hoạt, khi forward cần chia hết cho 4.
    inputs = tf.keras.Input((None, None, 3), name="noisy_rgb")

    # ENCODER: trích đặc trưng; hai lớp stride=2 giảm kích thước còn H/4 × W/4.
    conv1 = tf.keras.layers.Conv2D(
        filters[0], 3, strides=1, padding="same", activation="relu", name="conv_1"
    )(inputs)  # Mặc định 32 filters.
    conv2 = tf.keras.layers.Conv2D(
        filters[1], 3, strides=2, padding="same", activation="relu", name="conv_2"
    )(conv1)  # 32 filters, giảm H/W một nửa.
    conv3 = tf.keras.layers.Conv2D(
        filters[2], 3, strides=1, padding="same", activation="relu", name="conv_3"
    )(conv2)  # 64 filters.
    conv4 = tf.keras.layers.Conv2D(
        filters[3], 3, strides=2, padding="same", activation="relu", name="conv_4"
    )(conv3)  # 64 filters, giảm H/W thêm một nửa.

    # BOTTLENECK: biểu diễn đặc trưng gọn với 96 filters, không đổi kích thước.
    bottleneck = tf.keras.layers.Conv2D(
        filters[4], 3, strides=1, padding="same", activation="relu", name="bottleneck"
    )(conv4)

    # DECODER: khôi phục kích thước bằng nearest upsampling và học phần hiệu chỉnh.
    upsample1 = tf.keras.layers.UpSampling2D(
        2, interpolation="nearest", name="upsample_6"
    )(bottleneck)
    conv6 = tf.keras.layers.Conv2D(
        filters[5], 3, strides=1, padding="same", activation="relu", name="conv_6"
    )(upsample1)  # 64 filters.
    conv7 = tf.keras.layers.Conv2D(
        filters[6], 3, strides=1, padding="same", activation="relu", name="conv_7"
    )(conv6)  # 64 filters.
    upsample2 = tf.keras.layers.UpSampling2D(
        2, interpolation="nearest", name="upsample_8"
    )(conv7)
    conv8 = tf.keras.layers.Conv2D(
        filters[7], 3, strides=1, padding="same", activation="relu", name="conv_8"
    )(upsample2)  # 32 filters.
    conv9 = tf.keras.layers.Conv2D(
        filters[8], 3, strides=1, padding="same", activation="relu", name="conv_9"
    )(conv8)  # 32 filters.
    correction = tf.keras.layers.Conv2D(
        filters[9], 3, strides=1, padding="same", activation="linear", name="conv_10"
    )(conv9)  # 3 kênh RGB; linear cho phép hiệu chỉnh âm và dương.

    # GLOBAL RESIDUAL: ảnh khôi phục = ảnh noisy + phần hiệu chỉnh mạng học được.
    # Không clip khi train để giữ gradient; evaluate/inference clip về [0,1].
    output = tf.keras.layers.Add(name="residual_reconstruction")([inputs, correction])
    return tf.keras.Model(inputs, output, name="residual_lightweight_cae")



def build_skip_cae(filters=None):
    """Model B: identical convolution budget, additive H/2 and H encoder skips."""
    # Reuse the unchanged baseline layer declarations/configuration, but construct
    # a separate graph. No baseline model or checkpoint is modified.
    baseline = build_cae(filters)
    if baseline.get_layer("conv_3").filters != baseline.get_layer("conv_7").filters:
        raise ValueError("H/2 skip requires filters[2] == filters[6]")
    if baseline.get_layer("conv_1").filters != baseline.get_layer("conv_9").filters:
        raise ValueError("H skip requires filters[0] == filters[8]")
    inputs = tf.keras.Input((None, None, 3), name="noisy_rgb")
    x = inputs
    encoder = {}
    for layer in baseline.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer.name == "residual_reconstruction":
            x = tf.keras.layers.Add(name=layer.name)([inputs, x])
            continue
        x = layer.__class__.from_config(layer.get_config())(x)
        if layer.name in ("conv_1", "conv_3"):
            encoder[layer.name] = x
        if layer.name in ("conv_7", "conv_9"):
            source = encoder["conv_3" if layer.name == "conv_7" else "conv_1"]
            if tuple(source.shape) != tuple(x.shape):
                raise ValueError(f"Skip shape mismatch at {layer.name}")
            x = tf.keras.layers.Add(name="skip_half" if layer.name == "conv_7" else "skip_full")([x, source])
    return tf.keras.Model(inputs, x, name="skip_residual_cae")


HARDWARE_FILTERS = [16, 16, 32, 32, 48, 32, 32, 16, 16, 3]


def build_model(config):
    """Missing architecture preserves all legacy baseline YAML files."""
    architecture = config.get("architecture", "baseline")
    if architecture == "baseline":
        return build_cae(config.get("filters"))
    if architecture == "skip_cae":
        return build_skip_cae(config.get("filters"))
    if architecture == "hardware_cae":
        filters = config.get("filters", HARDWARE_FILTERS)
        if list(filters) != HARDWARE_FILTERS:
            raise ValueError("hardware_cae requires the fixed Model C filters")
        return build_cae(filters)
    if architecture == "paper_guided_cae":
        from .model_d import build_paper_guided_cae
        return build_paper_guided_cae(config.get("filters"))
    if architecture == "two_skip_cae":
        from .model_e import build_two_skip_cae
        return build_two_skip_cae(config.get("filters"))
    raise ValueError(f"Unknown model architecture: {architecture}")

def complexity(model, height=128, width=128):
    if height % 4 or width % 4:
        raise ValueError("Complexity input dimensions must be divisible by 4")
    h, w, channels = height, width, 3
    rows = []
    # Vòng lặp này chỉ đếm MACs của mạng đã tạo, không sinh lớp CNN.
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
    print_summary(build_model(load_config(args.config)["model"]))


if __name__ == "__main__":
    from .common import run_cli
    run_cli(main)
