"""Model E: Model D's seven convolutions with two additive encoder skips."""
import tensorflow as tf
from .model_d import FILTERS, STRIDES


def _skip_add(decoder, encoder, name):
    if tuple(decoder.shape) != tuple(encoder.shape):
        raise ValueError(f'{name}: encoder/decoder tensor shapes must match')
    return tf.keras.layers.Add(name=name)([decoder, encoder])


def build_two_skip_cae(filters=None):
    """RGB H/W divisible by four; only the two Add edges differ from Model D."""
    filters = tuple(FILTERS if filters is None else filters)
    if filters != FILTERS:
        raise ValueError(f'Model E uses fixed filter counts {list(FILTERS)}')
    noisy = tf.keras.Input((None, None, 3), name='noisy_rgb')
    x = noisy
    encoder = []
    for i in range(3):
        x = tf.keras.layers.Conv2D(
            filters[i], 3, strides=STRIDES[i], padding='same',
            activation='relu', use_bias=True, name=f'e_conv_{i+1}'
        )(x)
        if i < 2:
            encoder.append(x)

    x = tf.keras.layers.UpSampling2D(
        (2, 2), interpolation='nearest', name='e_upsample_1'
    )(x)
    x = tf.keras.layers.Conv2D(
        filters[3], 3, padding='same', activation='relu', use_bias=True, name='e_conv_4'
    )(x)
    x = _skip_add(x, encoder[1], 'e_skip_half')
    x = tf.keras.layers.Conv2D(
        filters[4], 3, padding='same', activation='relu', use_bias=True, name='e_conv_5'
    )(x)
    x = tf.keras.layers.UpSampling2D(
        (2, 2), interpolation='nearest', name='e_upsample_2'
    )(x)
    x = tf.keras.layers.Conv2D(
        filters[5], 3, padding='same', activation='relu', use_bias=True, name='e_conv_6'
    )(x)
    x = _skip_add(x, encoder[0], 'e_skip_full')
    correction = tf.keras.layers.Conv2D(
        filters[6], 3, padding='same', activation='linear', use_bias=True, name='e_conv_7'
    )(x)
    clean = tf.keras.layers.Add(name='e_residual_add')([noisy, correction])
    return tf.keras.Model(noisy, clean, name='two_skip_7conv_cae')
