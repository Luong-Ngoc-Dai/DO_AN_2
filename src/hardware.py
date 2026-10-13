"""Graph shape/layout metadata and analytical batch-one buffer estimates.

No synthesis, quantization, or TensorFlow allocator memory measurements.
"""
import math
import tensorflow as tf


PRECISION_BYTES = {'FP32': 4, 'INT16': 2, 'INT8': 1}


def storage(elements):
    return {precision: int(elements)*size for precision, size in PRECISION_BYTES.items()}


def describe_layers(model, height=128, width=128):
    if height <= 0 or width <= 0 or height % 4 or width % 4:
        raise ValueError('Input dimensions must be positive and divisible by 4')
    # Materialize a batch-one graph's intermediate outputs for shape inspection.
    values = tf.keras.Model(model.input, [layer.output for layer in model.layers])(
        tf.zeros((1, height, width, 3)), training=False)
    shapes = {layer.name: list(value.shape[1:]) for layer, value in zip(model.layers, values)}
    rows = []
    for layer in model.layers:
        incoming = [] if isinstance(layer, tf.keras.layers.InputLayer) else tf.nest.flatten(layer.input)
        sources = [tensor._keras_history.operation.name for tensor in incoming]
        row = {'name': layer.name, 'operation': layer.__class__.__name__,
               'sources': sources, 'input_shapes': [shapes[name] for name in sources],
               'output_shape': shapes[layer.name], 'output_elements': math.prod(shapes[layer.name])}
        if isinstance(layer, tf.keras.layers.Conv2D):
            input_shape = row['input_shapes'][0]
            padding = []
            for size, result, kernel, stride in zip(input_shape[:2], row['output_shape'][:2], layer.kernel_size, layer.strides):
                total = max((result-1)*stride+kernel-size, 0)
                padding.append([total//2, total-total//2])
            row.update(kernel_size=list(layer.kernel_size), strides=list(layer.strides),
                       padding=layer.padding, padding_top_bottom=padding[0], padding_left_right=padding[1],
                       input_channels=input_shape[2], output_channels=layer.filters,
                       activation=layer.activation.__name__, kernel_layout='HWIO',
                       parameters=layer.count_params())
        elif isinstance(layer, tf.keras.layers.UpSampling2D):
            row.update(scale=list(layer.size), interpolation=layer.interpolation)
        rows.append(row)
    return rows


def memory_report(model, layers):
    kernels = sum(math.prod(weight.shape) for layer in model.layers
                  if isinstance(layer, tf.keras.layers.Conv2D) for weight in [layer.kernel])
    biases = sum(math.prod(layer.bias.shape) for layer in model.layers
                 if isinstance(layer, tf.keras.layers.Conv2D) and layer.use_bias)
    largest = max(layers, key=lambda row: row['output_elements'])
    largest_conv = max((row for row in layers if row['operation'] == 'Conv2D'), key=lambda row: row['output_elements'])
    # Sequential, full-frame graph schedule, allocate output before releasing its
    # last-used sources. Global input/encoder skips remain live until their Add.
    last_use = {row['name']: index for index, row in enumerate(layers)}
    for index, row in enumerate(layers):
        for source in row['sources']:
            last_use[source] = index
    live, peak, timeline = {}, 0, []
    for index, row in enumerate(layers):
        live[row['name']] = row['output_elements']
        resident = sum(live.values())
        peak = max(peak, resident)
        timeline.append({'layer': row['name'], 'live_elements_before_release': resident})
        live = {name: count for name, count in live.items() if last_use[name] > index}
    return {
        'conv2d_count': sum(row['operation'] == 'Conv2D' for row in layers),
        'kernel_elements': kernels, 'bias_elements': biases,
        'weight_storage_bytes': storage(kernels), 'bias_storage_bytes': storage(biases),
        'parameter_storage_bytes': storage(model.count_params()),
        'largest_feature_map': {'layer': largest['name'], 'shape': largest['output_shape'],
                                'elements': largest['output_elements'], 'storage_bytes': storage(largest['output_elements'])},
        'largest_conv_feature_map': {'layer': largest_conv['name'], 'shape': largest_conv['output_shape'],
                                     'elements': largest_conv['output_elements'], 'storage_bytes': storage(largest_conv['output_elements'])},
        'intermediate_memory': {'peak_live_elements': peak, 'peak_live_storage_bytes': storage(peak),
                                'timeline': timeline,
                                'assumptions': 'Batch 1; sequential full-frame layer schedule; allocate output before releasing last-used inputs; no in-place Add/ReLU or fused upsample; retains noisy input and encoder skips until Add. Excludes weights, convolution line buffers, accumulators, DMA, alignment, training gradients and allocator workspace. INT storage assumes uniform hypothetical element widths, not a calibrated quantized implementation.'},
    }
