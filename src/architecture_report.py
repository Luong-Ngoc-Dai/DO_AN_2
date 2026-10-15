"""Export summaries and convolution costs without training or loading weights."""
import argparse
import io
from pathlib import Path
from .model import build_cae, build_skip_cae, complexity, HARDWARE_FILTERS
from .hardware import describe_layers, memory_report
from .common import save_json, run_cli


def report(output, height=128, width=128):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for label, builder in [('baseline', build_cae), ('skip_cae', build_skip_cae),
                           ('hardware_cae', lambda: build_cae(HARDWARE_FILTERS))]:
        model = builder()
        info = complexity(model, height, width)
        layers = describe_layers(model, height, width)
        info.update(memory_report(model, layers))
        additions = [{'layer': row['name'], 'shape': row['output_shape'],
                      'scalar_adds': row['output_elements']}
                     for row in layers if row['operation'] == 'Add']
        save_json(output/f'{label}_layers.json', layers)
        info['additions'] = additions
        info['scalar_adds'] = sum(item['scalar_adds'] for item in additions)
        info['conv_plus_add_flops'] = info['conv_flops'] + info['scalar_adds']
        buffer = io.StringIO()
        model.summary(print_fn=lambda line: buffer.write(line+'\n'))
        (output/f'{label}_summary.txt').write_text(buffer.getvalue(), encoding='utf-8')
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
