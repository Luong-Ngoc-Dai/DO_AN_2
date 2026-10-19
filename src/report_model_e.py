"""Compute convolution, Add and skip-buffer costs from the actual Model E graph."""
import argparse
from pathlib import Path
from .common import load_config, save_json, run_cli
from .model import build_model, complexity
from .hardware import describe_layers, memory_report, storage


def report_model_e(config='configs/model_e.yaml', output='runs/model_e_architecture'):
    cfg = load_config(config)
    if cfg['model'].get('architecture') != 'two_skip_cae':
        raise ValueError('report_model_e requires a two_skip_cae config')
    model = build_model(cfg['model'])
    layers = describe_layers(model, 128, 128)
    info = complexity(model)
    info.update(memory_report(model, layers))
    additions, features = [], []
    for row in layers:
        if row['operation'] != 'Add':
            continue
        additions.append({'name': row['name'], 'sources': row['sources'],
                          'shape': row['output_shape'], 'scalar_adds': row['output_elements']})
        if row['name'] in ('e_skip_half', 'e_skip_full'):
            features.append({'source': row['sources'][1], 'used_by': row['name'],
                             'shape': row['input_shapes'][1], 'elements': row['output_elements'],
                             'storage_bytes': storage(row['output_elements'])})
    info['additions'] = additions
    info['scalar_adds'] = sum(row['scalar_adds'] for row in additions)
    info['extra_skip_scalar_adds'] = sum(row['elements'] for row in features)
    info['conv_plus_add_flops'] = info['conv_flops']+info['scalar_adds']
    info['skip_feature_maps'] = features
    info['skip_feature_storage_bytes'] = storage(sum(row['elements'] for row in features))
    info['skip_storage_note'] = 'Sum of two retained full-frame encoder maps, batch 1; not total FPGA memory. INT8 storage is hypothetical; no quantization is performed.'
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/'complexity.json', info)
    save_json(out/'layers.json', layers)
    with (out/'model_summary.txt').open('w', encoding='utf-8') as stream:
        model.summary(print_fn=lambda line: stream.write(line+'\n'))
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/model_e.yaml')
    parser.add_argument('--output', default='runs/model_e_architecture')
    args = parser.parse_args()
    info = report_model_e(args.config, args.output)
    for key in ('conv2d_count', 'parameters', 'conv_macs', 'conv_flops', 'extra_skip_scalar_adds', 'skip_feature_storage_bytes'):
        print(f'{key}: {info[key]}')
    print('Saved report:', args.output)


if __name__ == '__main__':
    run_cli(main)
