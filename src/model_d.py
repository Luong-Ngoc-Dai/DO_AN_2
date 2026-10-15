"""Model D: 7-Conv literature-guided denoising CAE for SIDD / Verilog FPGA research.

Literature *inspiration*, not a reimplementation:
- Isik et al. (2023): 3 encoder + 4 decoder convolutions in a CAE FPGA study.
- Yang et al. (2023) PSL-AE: RGB 3x3 convolution and progressively increased
  encoder channels, studied on SIDD.

Inference graph only: Conv2D, ReLU, nearest upsampling, residual addition.
Input height and width must be multiples of 4 for equal output shape.
"""
import tensorflow as tf

FILTERS = (16, 32, 48, 32, 16, 16, 3)
STRIDES = (1, 2, 2, 1, 1, 1, 1)


def build_paper_guided_cae(filters=None):
    """Build a 7-convolution full-RGB residual CAE, with no pretrained weights."""
    filters = tuple(FILTERS if filters is None else filters)
    if filters != FILTERS:
        raise ValueError(f"Model D uses fixed filter counts {list(FILTERS)}")

    noisy = tf.keras.Input(shape=(None, None, 3), name="noisy_rgb")
    x = noisy
    for i in range(3):
        x = tf.keras.layers.Conv2D(
            filters[i], kernel_size=3, strides=STRIDES[i], padding="same",
            activation="relu", use_bias=True, name=f"d_conv_{i+1}"
        )(x)

    x = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation="nearest", name="d_upsample_1"
    )(x)
    for i in (3, 4):
        x = tf.keras.layers.Conv2D(
            filters[i], kernel_size=3, strides=1, padding="same",
            activation="relu", use_bias=True, name=f"d_conv_{i+1}"
        )(x)

    x = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation="nearest", name="d_upsample_2"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters[5], kernel_size=3, strides=1, padding="same",
        activation="relu", use_bias=True, name="d_conv_6"
    )(x)
    correction = tf.keras.layers.Conv2D(
        filters[6], kernel_size=3, strides=1, padding="same",
        activation="linear", use_bias=True, name="d_conv_7"
    )(x)
    clean = tf.keras.layers.Add(name="d_residual_add")([noisy, correction])
    return tf.keras.Model(noisy, clean, name="paper_guided_7conv_cae")
