"""Export exact Model E FP32 kernels/biases and additive graph wiring as NumPy."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import tensorflow as tf
from .common import run_cli, save_json
from .export_model_d import sha256
from .hardware import describe_layers
from .model_e import build_two_skip_cae


def export_model_e(model_path, output):
    path, output = Path(model_path), Path(output)
    if output.exists():
        raise FileExistsError(f'Export directory already exists: {output}; choose a new directory')
    source_hash = sha256(path)
    model = tf.keras.models.load_model(path, compile=False)
    canonical = lambda graph: json.loads(json.dumps(graph.get_config()))
    if canonical(model) != canonical(build_two_skip_cae()):
        raise ValueError('Checkpoint is not the expected Model E graph')
    convolutions = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    tensors = [layer.get_weights() for layer in convolutions]
    if any(array.dtype != np.float32 or not np.all(np.isfinite(array))
           for pair in tensors for array in pair):
        raise ValueError('Model E export requires finite FP32 kernels and biases')
    layers = describe_layers(model, 128, 128)
    additions = []
    for row in layers:
        if row['operation'] == 'Add':
            additions.append({'name': row['name'], 'sources': row['sources'],
                              'consumers': [other['name'] for other in layers if row['name'] in other['sources']],
                              'input_shapes_at_reference': row['input_shapes'],
                              'output_shape_at_reference': row['output_shape']})
    manifest = {'version': 1, 'model': model.name, 'architecture': 'two_skip_cae',
                'checkpoint_file': path.name, 'checkpoint_sha256': source_hash,
                'input_layout': 'NHWC', 'kernel_layout': 'HWIO', 'flatten_order': 'C',
                'kernel_flat_address': '(((kh * Kw + kw) * Cin + ci) * Cout + co)',
                'dtype': 'float32', 'quantized': False, 'reference_input_shape': [128, 128, 3],
                'parameters': model.count_params(),
                'skip_connections': [row for row in additions if row['name'] in ('e_skip_half', 'e_skip_full')],
                'global_residual': next(row for row in additions if row['name'] == 'e_residual_add'),
                'note': 'TensorFlow cross-correlation; do not flip kernels. Skip Adds follow Conv4/Conv6 ReLU, before Conv5/Conv7. No quantization or teacher.',
                'layers': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.model-e-export-', dir=output.parent) as temporary:
        staging = Path(temporary)/'export'
        staging.mkdir()
        for index, (layer, (kernel, bias)) in enumerate(zip(convolutions, tensors), 1):
            base = f'conv_{index:02d}'
            kernel_path, bias_path = staging/f'{base}_kernel.npy', staging/f'{base}_bias.npy'
            np.save(kernel_path, kernel, allow_pickle=False)
            np.save(bias_path, bias, allow_pickle=False)
            manifest['layers'].append({'index': index, 'name': layer.name,
                                       'kernel_shape': list(kernel.shape), 'bias_shape': list(bias.shape),
                                       'kernel_npy': kernel_path.name, 'bias_npy': bias_path.name,
                                       'kernel_sha256': sha256(kernel_path), 'bias_sha256': sha256(bias_path)})
        save_json(staging/'manifest.json', manifest)
        if sha256(path) != source_hash:
            raise ValueError('Checkpoint changed during export; retry with a stable checkpoint')
        staging.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print('Saved FP32 Model E tensors:', export_model_e(args.model, args.output))


if __name__ == '__main__':
    run_cli(main)
