"""Export a Model C .keras checkpoint as exact FP32 NumPy and decimal CSV.

No retraining, scaling, integer casting or quantization is performed. NumPy files
are the authoritative tensor representation; CSV row addresses use C-order HWIO.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import tensorflow as tf
from .common import save_json, run_cli
from .hardware import describe_layers
from .model import build_cae, HARDWARE_FILTERS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export_weights(checkpoint, output, height=128, width=128):
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise ValueError(f'Export directory already exists: {output}; choose a new directory')
    source_hash = sha256(checkpoint)
    model = tf.keras.models.load_model(checkpoint, compile=False)
    expected = build_cae(HARDWARE_FILTERS)
    # Compare the entire serialized graph, including global Add and absence of
    # encoder skips. Filter counts alone would also accept a small Model B.
    canonical = lambda item: json.loads(json.dumps(item))
    if canonical(model.get_config()) != canonical(expected.get_config()):
        raise ValueError('Checkpoint is not the fixed Model C graph')
    layers = describe_layers(model, height, width)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.model-c-export-', dir=output.parent) as temporary:
        staging = Path(temporary)/'export'
        staging.mkdir()
        manifest = {'version': 1, 'architecture': 'hardware_cae', 'model_name': model.name,
                    'checkpoint_file': checkpoint.name, 'checkpoint_sha256': source_hash,
                    'input_shape': [height, width, 3], 'tensor_layout': 'NHWC',
                    'kernel_layout': 'HWIO (kernel_height, kernel_width, input_channel, output_channel)',
                    'kernel_flat_address': '(((kh * Kw + kw) * Cin + ci) * Cout + co)',
                    'bias_layout': 'O (output_channel)', 'flatten_order': 'C',
                    'numeric_format': 'FP32, unquantized; CSV uses 9 significant decimal digits',
                    'convolution': 'TensorFlow cross-correlation; do not flip spatial kernel axes',
                    'training_status': 'User-supplied checkpoint; export does not certify training or quality',
                    'layers': []}
        conv_index = 0
        for layer in model.layers:
            if not isinstance(layer, tf.keras.layers.Conv2D):
                continue
            conv_index += 1
            descriptor = {'conv_index': conv_index, 'name': layer.name, 'tensors': {}}
            for name, array in zip(('kernel', 'bias'), layer.get_weights()):
                if array.dtype != np.float32 or not np.all(np.isfinite(array)):
                    raise ValueError(f'{layer.name}/{name}: expected finite FP32 values')
                stem = f'{conv_index:02d}_{layer.name}_{name}'
                npy, csv = staging/f'{stem}.npy', staging/f'{stem}.csv'
                np.save(npy, array, allow_pickle=False)
                np.savetxt(csv, array.reshape(-1, order='C'), fmt='%.9g', header='value_fp32', comments='')
                descriptor['tensors'][name] = {'shape': list(array.shape), 'dtype': 'float32',
                                              'elements': array.size, 'npy': npy.name, 'csv': csv.name,
                                              'npy_sha256': sha256(npy), 'csv_sha256': sha256(csv)}
            manifest['layers'].append(descriptor)
        save_json(staging/'manifest.json', manifest)
        save_json(staging/'layers.json', layers)
        if sha256(checkpoint) != source_hash:
            raise ValueError('Checkpoint changed during export; retry with a stable checkpoint')
        staging.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model C best.keras or another compatible full checkpoint')
    parser.add_argument('--output', required=True, help='New directory; existing exports are not overwritten')
    args = parser.parse_args()
    print('Exported FP32 Model C tensors:', export_weights(args.model, args.output))


if __name__ == '__main__':
    run_cli(main)
