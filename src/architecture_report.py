"""Export summaries and convolution costs without training or loading weights."""
import argparse
import io
from pathlib import Path
import tensorflow as tf
from .model import build_cae, build_skip_cae, complexity
from .common import save_json, run_cli


def report(output, height=128, width=128):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for label, builder in [('baseline', build_cae), ('skip_cae', build_skip_cae)]:
        model = builder()
        info = complexity(model, height, width)
        # Inspect concrete graph shapes without executing training.
        shapes = tf.keras.Model(model.input, [l.output for l in model.layers])(tf.zeros((1, height, width, 3)))
        additions = []
        for layer, tensor in zip(model.layers, shapes):
            if isinstance(layer, tf.keras.layers.Add):
                count = int(tensor.shape[1]*tensor.shape[2]*tensor.shape[3])
                additions.append({'layer': layer.name, 'shape': list(tensor.shape[1:]), 'scalar_adds': count})
        info['additions'] = additions
        info['scalar_adds'] = sum(item['scalar_adds'] for item in additions)
        info['conv_plus_add_flops'] = info['conv_flops'] + info['scalar_adds']
        buffer = io.StringIO()
        model.summary(print_fn=lambda line: buffer.write(line+'\n'))
        (output/f'{label}_summary.txt').write_text(buffer.getvalue())
        rows[label] = info
    save_json(output/'comparison.json', rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/architecture_comparison')
    args = parser.parse_args()
    rows = report(args.output)
    for name, row in rows.items():
        print(name, row['parameters'], 'parameters;', row['conv_macs'], 'conv MACs;', row['conv_flops'], 'conv FLOPs;', row['scalar_adds'], 'scalar adds')


if __name__ == '__main__':
    run_cli(main)
