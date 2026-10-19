"""Two-skip D/E ablation, export and shared training/inference integration."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from src.common import load_config
from src.model import build_model, complexity
from src.model_d import build_paper_guided_cae, FILTERS, STRIDES
from src.model_e import build_two_skip_cae
from src.export_model_e import export_model_e, sha256


def test_e_matches_d_convolutions_and_training_config():
    d_cfg = load_config('configs/model_d.yaml')
    e_cfg = load_config('configs/model_e.yaml')
    assert e_cfg['data'] == d_cfg['data']
    assert e_cfg['seed'] == d_cfg['seed'] == 42
    assert e_cfg['training']['output'] == 'runs/model_e_100epochs'
    assert {k: v for k, v in e_cfg['training'].items() if k != 'output'} == {k: v for k, v in d_cfg['training'].items() if k != 'output'}
    tf.keras.utils.set_random_seed(42)
    d = build_paper_guided_cae()
    tf.keras.utils.set_random_seed(42)
    e = build_model(e_cfg['model'])
    d_convs = [layer for layer in d.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    e_convs = [layer for layer in e.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    assert len(e_convs) == 7
    assert tuple(layer.filters for layer in e_convs) == FILTERS
    assert [layer.strides for layer in e_convs] == [(s, s) for s in STRIDES]
    for original, candidate in zip(d_convs, e_convs):
        d_config, e_config = original.get_config(), candidate.get_config()
        d_config.pop('name')
        e_config.pop('name')
        assert d_config == e_config
        for a, b in zip(original.get_weights(), candidate.get_weights()):
            np.testing.assert_array_equal(a, b)
    assert sum(isinstance(layer, tf.keras.layers.Add) for layer in e.layers) == 3
    upsampling = [layer for layer in e.layers if isinstance(layer, tf.keras.layers.UpSampling2D)]
    assert len(upsampling) == 2 and all(layer.size == (2, 2) and layer.interpolation == 'nearest' for layer in upsampling)
    assert all(isinstance(layer, (tf.keras.layers.InputLayer, tf.keras.layers.Conv2D,
                                 tf.keras.layers.Add, tf.keras.layers.UpSampling2D)) for layer in e.layers)
    assert complexity(e) == {**complexity(d), 'layers': [{**row, 'layer': row['layer'].replace('d_', 'e_')} for row in complexity(d)['layers']]}
    assert sum(np.prod(weight.shape) for weight in e.trainable_weights) == 40195
    with pytest.raises(ValueError, match='fixed filter'):
        build_two_skip_cae([8, 16, 24, 16, 8, 8, 3])


@pytest.mark.parametrize('hw', [(128, 128), (128, 256)])
def test_e_skip_wiring_shapes_additions_and_identity(hw):
    model = build_two_skip_cae()
    x = tf.random.uniform((1, *hw, 3))
    result = model(x)
    assert result.shape == x.shape and bool(tf.reduce_all(tf.math.is_finite(result)))
    for name, sources, consumer, expected in (
        ('e_skip_half', ['e_conv_4', 'e_conv_2'], 'e_conv_5', (1, hw[0]//2, hw[1]//2, 32)),
        ('e_skip_full', ['e_conv_6', 'e_conv_1'], 'e_conv_7', (1, *hw, 16)),
    ):
        skip = model.get_layer(name)
        assert [tensor._keras_history.operation.name for tensor in skip.input] == sources
        assert model.get_layer(consumer).input._keras_history.operation.name == name
        a, b, y = tf.keras.Model(model.input, [*skip.input, skip.output])(x)
        assert a.shape == b.shape == y.shape == expected
        np.testing.assert_array_equal(y, a+b)
    assert [tensor._keras_history.operation.name for tensor in model.get_layer('e_residual_add').input] == ['noisy_rgb', 'e_conv_7']
    last = model.get_layer('e_conv_7')
    last.set_weights([np.zeros_like(weight) for weight in last.get_weights()])
    np.testing.assert_array_equal(model(x), x)


def test_e_cost_report(tmp_path, monkeypatch):
    from src.report_model_e import report_model_e
    original_open = Path.open
    def windows_open(path, mode='r', *args, **kwargs):
        if path.suffix == '.txt' and 'w' in mode:
            kwargs.setdefault('encoding', 'cp1252')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', windows_open)
    info = report_model_e(output=tmp_path/'report')
    sizes, inputs = [128, 64, 32, 64, 64, 128, 128], [3]+list(FILTERS[:-1])
    assert info['parameters'] == sum(9*ci*co+co for ci, co in zip(inputs, FILTERS)) == 40195
    assert info['conv_macs'] == sum(s*s*9*ci*co for s, ci, co in zip(sizes, inputs, FILTERS)) == 160432128
    assert info['conv_flops'] == 320864256
    assert info['conv2d_count'] == 7
    assert info['extra_skip_scalar_adds'] == 64*64*32+128*128*16 == 393216
    assert info['scalar_adds'] == 442368
    assert info['skip_feature_storage_bytes']['FP32'] == 1572864
    assert info['skip_feature_storage_bytes']['INT8'] == 393216
    assert info['skip_feature_maps'][0]['shape'] == [64, 64, 32]
    assert info['skip_feature_maps'][1]['shape'] == [128, 128, 16]
    assert 'e_skip_half' in (tmp_path/'report'/'model_summary.txt').read_text(encoding='utf-8')
    assert json.loads((tmp_path/'report'/'complexity.json').read_text()) == info


def test_e_checkpoint_export_roundtrip(tmp_path):
    model = build_two_skip_cae()
    checkpoint = tmp_path/'best.keras'
    model.save(checkpoint)
    restored = tf.keras.models.load_model(checkpoint, compile=False)
    x = tf.random.uniform((1, 16, 24, 3))
    np.testing.assert_array_equal(model(x), restored(x))
    original_hash = sha256(checkpoint)
    output = export_model_e(checkpoint, tmp_path/'export')
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['checkpoint_sha256'] == original_hash == sha256(checkpoint)
    assert manifest['kernel_layout'] == 'HWIO' and manifest['flatten_order'] == 'C'
    assert not manifest['quantized'] and len(manifest['layers']) == 7
    assert [row['sources'] for row in manifest['skip_connections']] == [['e_conv_4', 'e_conv_2'], ['e_conv_6', 'e_conv_1']]
    assert [row['consumers'] for row in manifest['skip_connections']] == [['e_conv_5'], ['e_conv_7']]
    reconstructed = build_two_skip_cae()
    source_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    target_layers = [layer for layer in reconstructed.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    for index, (source, target, row) in enumerate(zip(source_layers, target_layers, manifest['layers']), 1):
        assert row['index'] == index and row['name'] == source.name
        arrays = [np.load(output/row[key], allow_pickle=False) for key in ('kernel_npy', 'bias_npy')]
        assert sha256(output/row['kernel_npy']) == row['kernel_sha256']
        assert sha256(output/row['bias_npy']) == row['bias_sha256']
        for actual, expected in zip(arrays, source.get_weights()):
            np.testing.assert_array_equal(actual, expected)
        target.set_weights(arrays)
    np.testing.assert_array_equal(model(x), reconstructed(x))
    with pytest.raises(FileExistsError):
        export_model_e(checkpoint, output)


@pytest.mark.parametrize('kind', ['d', 'changed_e'])
def test_e_export_rejects_incompatible_checkpoint(tmp_path, kind):
    if kind == 'd':
        model = build_paper_guided_cae()
    else:
        config = build_two_skip_cae().get_config()
        for layer in config['layers']:
            if layer['config']['name'] == 'e_upsample_1':
                layer['config']['interpolation'] = 'bilinear'
        model = tf.keras.Model.from_config(config)
    checkpoint = tmp_path/'other.keras'
    model.save(checkpoint)
    with pytest.raises(ValueError, match='expected Model E graph'):
        export_model_e(checkpoint, tmp_path/'export')
    assert not (tmp_path/'export').exists()


def test_e_smoke_evaluation_full_image_and_export(tmp_path, monkeypatch):
    from test_pipeline import fake_sidd
    from src.dataset import discover_pairs, split_pairs, save_split, read_pair
    from src.train import train_experiment
    from src.checkpoint import load_recovery
    from src.evaluate import evaluate
    from src.inference import tiled_denoise, comparison_report
    original_open = Path.open
    def windows_open(path, mode='r', *args, **kwargs):
        if path.name == 'model_summary.txt' and 'w' in mode:
            kwargs.setdefault('encoding', 'cp1252')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', windows_open)
    root = tmp_path/'data'
    fake_sidd(root)
    pairs = discover_pairs(root)
    train, val = split_pairs(pairs, seed=42)
    split = tmp_path/'sidd_split.json'
    save_split(split, root, train, val, 42)
    split_bytes = split.read_bytes()
    cfg = load_config('configs/model_e.yaml')
    cfg['data'].update(root=str(root), split=str(split), expected_pairs=10)
    cfg['training']['output'] = str(tmp_path/'run')
    output = train_experiment(cfg, smoke=True)
    assert split.read_bytes() == split_bytes
    for name in ('best.keras', 'restored_best.keras', 'training_history.csv', 'training_summary.json',
                 'quality_history.png', 'model_summary.txt', 'complexity.json'):
        assert (output/name).is_file()
    assert 'e_skip_full' in (output/'model_summary.txt').read_text(encoding='utf-8')
    summary = json.loads((output/'training_summary.json').read_text())
    assert summary['smoke'] and summary['epochs_executed'] == 1
    rows = list(csv.DictReader((output/'training_history.csv').open(encoding='utf-8')))
    assert all(np.isfinite(float(rows[0][key])) for key in ('loss', 'val_psnr', 'val_ssim', 'val_uqi'))
    recovered, state = load_recovery(output)
    assert state['epoch'] == 1 and int(recovered.optimizer.iterations) == 1
    model = tf.keras.models.load_model(output/'best.keras', compile=False)
    result = evaluate(model, root, val[:1], tmp_path/'evaluation', tile_size=128, overlap=32)
    assert result['image_count'] == 1 and all(np.isfinite(result['means'][key]) for key in ('psnr_denoised', 'ssim_denoised', 'uqi_denoised'))
    noisy, gt = read_pair(root, val[0])
    denoised = tiled_denoise(model, noisy, tile_size=128, overlap=32)
    assert noisy.shape == denoised.shape == gt.shape
    comparison_report(noisy, denoised, gt, tmp_path/'evaluation')
    assert (tmp_path/'evaluation'/'comparison.png').is_file()
    export_model_e(output/'best.keras', tmp_path/'trained-export')


def test_e_interrupted_training_resume(tmp_path, monkeypatch):
    from test_pipeline import fake_sidd
    from src.dataset import discover_pairs, split_pairs, save_split
    from src.checkpoint import Recovery, load_recovery
    from src.train import train_experiment
    root = tmp_path/'data'
    fake_sidd(root)
    train, val = split_pairs(discover_pairs(root), seed=42)
    split = tmp_path/'sidd_split.json'
    save_split(split, root, train, val, 42)
    split_bytes = split.read_bytes()
    cfg = load_config('configs/model_e.yaml')
    cfg['data'].update(root=str(root), split=str(split), expected_pairs=10, patch_size=16, patches_per_image=1)
    cfg['training'].update(output=str(tmp_path/'run'), epochs=2)
    original = Recovery.on_epoch_end
    before = {}
    def interrupt(self, epoch, logs=None):
        original(self, epoch, logs)
        before['optimizer'] = [var.numpy().copy() for var in self.model.optimizer.variables]
        before['weights'] = [var.numpy().copy() for var in self.model.weights]
        raise RuntimeError('simulated interruption')
    monkeypatch.setattr(Recovery, 'on_epoch_end', interrupt)
    with pytest.raises(RuntimeError, match='simulated'):
        train_experiment(cfg)
    restored, state = load_recovery(tmp_path/'run')
    assert state['epoch'] == 1
    for actual, expected in zip(restored.optimizer.variables, before['optimizer']):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(restored.weights, before['weights']):
        np.testing.assert_array_equal(actual, expected)
    monkeypatch.setattr(Recovery, 'on_epoch_end', original)
    output = train_experiment(cfg, resume=True)
    model, state = load_recovery(output)
    assert state['epoch'] == 2 and int(model.optimizer.iterations) == 2
    assert split.read_bytes() == split_bytes
    assert [row['epoch'] for row in csv.DictReader((output/'training_history.csv').open())] == ['0', '1']


def test_e_requires_original_split_and_counts(tmp_path, monkeypatch):
    from src.dataset import Pair, save_split
    from src.train import train_experiment
    import src.train as train_module
    cfg = load_config('configs/model_e.yaml')
    pairs = [Pair(f'scene_{i:03}', str(j), f'scene_{i:03}/NOISY_SRGB_{j:03}.PNG',
                  f'scene_{i:03}/GT_SRGB_{j:03}.PNG') for i in range(160) for j in (10, 11)]
    monkeypatch.setattr(train_module, 'discover_pairs', lambda *args: pairs)
    cfg['data'].pop('split')
    with pytest.raises(ValueError, match='Model E requires an existing shared split'):
        train_experiment(cfg)
    split = tmp_path/'wrong.json'
    save_split(split, tmp_path, pairs[:254], pairs[254:], 42)
    cfg['data']['split'] = str(split)
    split_bytes = split.read_bytes()
    with pytest.raises(ValueError, match='Model E requires the original 256/64'):
        train_experiment(cfg)
    assert split.read_bytes() == split_bytes
