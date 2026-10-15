"""Report analytical Model D complexity and batch-one activation estimates."""
import argparse
from pathlib import Path
from .common import load_config, save_json, run_cli
from .model import build_model, complexity
from .hardware import describe_layers, memory_report


def report_model_d(config="configs/model_d.yaml", output="runs/model_d_architecture"):
    model = build_model(load_config(config)["model"])
    if model.name != "paper_guided_7conv_cae":
        raise ValueError("report_model_d requires a paper_guided_cae config")
    out = Path(output)
    out.mkdir(parents=True,exist_ok=True)
    layers = describe_layers(model, 128, 128)
    values = complexity(model)
    values.update(memory_report(model,layers))
    save_json(out/"complexity.json", values)
    save_json(out/"layers.json", layers)
    with (out/"model_summary.txt").open("w",encoding="utf-8") as f:
        model.summary(print_fn=lambda s: f.write(s+"\n"))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/model_d.yaml')
    parser.add_argument('--output', default='runs/model_d_architecture')
    a = parser.parse_args()
    values = report_model_d(a.config, a.output)
    print('Conv layers:', values['conv2d_count'])
    print('Parameters:', values['parameters'])
    print('Conv MACs:', values['conv_macs'])
    print('Conv FLOPs:', values['conv_flops'])
    print('Peak live bytes FP32 (estimated):',values['intermediate_memory']['peak_live_storage_bytes']['FP32'])
    print('Saved report:',a.output)

if __name__=='__main__':
    run_cli(main)
