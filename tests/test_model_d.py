"""Fast architecture / forwarding tests for Model D; no SIDD dataset required."""
import numpy as np
import pytest
import tensorflow as tf
from src.model import build_model, complexity
from src.common import load_config
from src.model_d import FILTERS


def test_d_structure_and_costs():
    cfg = load_config("configs/model_d.yaml")
    model = build_model(cfg["model"])
    convs = [x for x in model.layers if isinstance(x, tf.keras.layers.Conv2D)]
    ups = [x for x in model.layers if isinstance(x, tf.keras.layers.UpSampling2D)]
    adds = [x for x in model.layers if isinstance(x, tf.keras.layers.Add)]
    assert len(convs) == 7
    assert tuple(x.filters for x in convs) == FILTERS
    assert [x.strides for x in convs] == [(1,1),(2,2),(2,2),(1,1),(1,1),(1,1),(1,1)]
    assert len(ups) == 2 and all(x.interpolation == "nearest" for x in ups)
    assert len(adds) == 1
    assert all(x.kernel_size == (3,3) and x.padding == "same" for x in convs)
    assert [x.activation.__name__ for x in convs] == ["relu"]*6+["linear"]
    assert all(isinstance(x,(tf.keras.layers.InputLayer,tf.keras.layers.Conv2D,
                             tf.keras.layers.UpSampling2D,tf.keras.layers.Add)) for x in model.layers)
    info = complexity(model)
    assert info["parameters"] == 40195
    assert info["conv_macs"] == 160432128
    assert info["conv_flops"] == 320864256


@pytest.mark.parametrize("hw", [16, 32, 128])
def test_d_forward_identity_sanity(hw):
    cfg = load_config("configs/model_d.yaml")
    model = build_model(cfg["model"])
    x = tf.random.uniform((1, hw, hw, 3), seed=33)
    y = model(x, training=False)
    assert tuple(y.shape) == tuple(x.shape)
    assert bool(tf.reduce_all(tf.math.is_finite(y)))
    last = model.get_layer("d_conv_7")
    last.set_weights([np.zeros_like(w) for w in last.get_weights()])
    np.testing.assert_array_equal(model(x,training=False).numpy(), x.numpy())


def test_d_filters_fixed():
    cfg = load_config("configs/model_d.yaml")
    cfg["model"]["filters"] = [8,16,24,16,8,8,3]
    with pytest.raises(ValueError,match="fixed filter"):
        build_model(cfg["model"])


def test_d_config_and_independent_arithmetic():
    cfg = load_config('configs/model_d.yaml')
    assert cfg['seed'] == 42
    assert cfg['data']['split'] == 'runs/sidd_split.json'
    assert cfg['data']['expected_pairs'] == 320
    assert cfg['data']['patch_size'] == 128 and cfg['data']['patches_per_image'] == 4
    assert cfg['training']['batch_size'] == 8 and cfg['training']['epochs'] == 100
    assert cfg['training']['learning_rate'] == .0001
    sizes = [128, 64, 32, 64, 64, 128, 128]
    input_channels = [3]+list(FILTERS[:-1])
    model = build_model(cfg['model'])
    info = complexity(model)
    assert info['parameters'] == sum(9*ci*co+co for ci, co in zip(input_channels, FILTERS))
    assert info['conv_macs'] == sum(hw*hw*9*ci*co for hw, ci, co in zip(sizes, input_channels, FILTERS))
    assert model(np.zeros((1, 128, 256, 3), dtype=np.float32)).shape == (1, 128, 256, 3)


def test_d_report_and_export_roundtrip(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from src.report_model_d import report_model_d
    from src.export_model_d import export_model_d, sha256
    # Model summaries contain box-drawing characters, unencodable in CP1252.
    # Emulate a Windows default when a caller omits an explicit encoding.
    original_open = Path.open
    def windows_open(path, mode='r', *args, **kwargs):
        if path.suffix == '.txt' and 'w' in mode:
            kwargs.setdefault('encoding', 'cp1252')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', windows_open)
    values = report_model_d(output=tmp_path/'report')
    assert values['conv2d_count'] == 7 and values['parameters'] == 40195
    text = (tmp_path/'report'/'model_summary.txt').read_text(encoding='utf-8')
    assert 'd_residual_add' in text and any(ord(char) > 127 for char in text)
    tf.keras.utils.set_random_seed(61)
    model = build_model(load_config('configs/model_d.yaml')['model'])
    path = tmp_path/'best.keras'
    model.save(path)
    original_hash = sha256(path)
    restored = tf.keras.models.load_model(path, compile=False)
    x = tf.random.uniform((1, 16, 24, 3))
    np.testing.assert_array_equal(model(x), restored(x))
    output = export_model_d(path, tmp_path/'export')
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['checkpoint_sha256'] == original_hash == sha256(path)
    assert manifest['kernel_layout'] == 'HWIO' and not manifest['quantized']
    convs = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    target = build_model(load_config('configs/model_d.yaml')['model'])
    target_convs = [layer for layer in target.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    for index, (layer, target_layer, row) in enumerate(zip(convs, target_convs, manifest['layers']), 1):
        assert row['index'] == index and row['name'] == layer.name
        kernel = np.load(output/row['kernel_npy'], allow_pickle=False)
        bias = np.load(output/row['bias_npy'], allow_pickle=False)
        assert sha256(output/row['kernel_npy']) == row['kernel_sha256']
        assert sha256(output/row['bias_npy']) == row['bias_sha256']
        for actual, expected in zip((kernel, bias), layer.get_weights()):
            np.testing.assert_array_equal(actual, expected)
        target_layer.set_weights([kernel, bias])
    np.testing.assert_array_equal(model(x), target(x))
    with pytest.raises(FileExistsError):
        export_model_d(path, output)


@pytest.mark.parametrize('alteration', ['activation', 'upsampling'])
def test_d_export_rejects_same_cost_altered_graph(tmp_path, alteration):
    from src.export_model_d import export_model_d
    model = build_model(load_config('configs/model_d.yaml')['model'])
    config = model.get_config()
    for layer in config['layers']:
        if alteration == 'activation' and layer['config']['name'] == 'd_conv_7':
            layer['config']['activation'] = 'relu'
        if alteration == 'upsampling' and layer['config']['name'] == 'd_upsample_1':
            layer['config']['interpolation'] = 'bilinear'
    modified = tf.keras.Model.from_config(config)
    assert modified.count_params() == model.count_params()
    path = tmp_path/'altered.keras'
    modified.save(path)
    with pytest.raises(ValueError, match='expected Model D graph'):
        export_model_d(path, tmp_path/'export')
    assert not (tmp_path/'export').exists()


def test_d_smoke_training_evaluation_and_export(tmp_path, monkeypatch):
    import csv
    import json
    from pathlib import Path
    from test_pipeline import fake_sidd
    from src.dataset import discover_pairs, split_pairs, save_split
    from src.train import train_experiment
    from src.checkpoint import load_recovery
    from src.evaluate import evaluate
    from src.export_model_d import export_model_d
    original_open = Path.open
    def windows_open(path, mode='r', *args, **kwargs):
        if path.name == 'model_summary.txt' and 'w' in mode:
            kwargs.setdefault('encoding', 'cp1252')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', windows_open)
    root = tmp_path/'data'
    fake_sidd(root)
    train, val = split_pairs(discover_pairs(root), seed=42)
    split = tmp_path/'sidd_split.json'
    save_split(split, root, train, val, 42)
    before = split.read_bytes()
    cfg = load_config('configs/model_d.yaml')
    cfg['data'].update(root=str(root), split=str(split), expected_pairs=10)
    cfg['training']['output'] = str(tmp_path/'run')
    output = train_experiment(cfg, smoke=True)
    assert split.read_bytes() == before
    assert json.loads((output/'training_summary.json').read_text())['epochs_executed'] == 1
    for name in ('best.keras', 'restored_best.keras', 'training_history.csv', 'quality_history.png'):
        assert (output/name).is_file()
    assert 'd_conv_7' in (output/'model_summary.txt').read_text(encoding='utf-8')
    rows = list(csv.DictReader((output/'training_history.csv').open(encoding='utf-8')))
    assert all(np.isfinite(float(rows[0][key])) for key in ('val_psnr', 'val_ssim', 'val_uqi'))
    model, state = load_recovery(output)
    assert state['epoch'] == 1 and int(model.optimizer.iterations) == 1
    best = tf.keras.models.load_model(output/'best.keras', compile=False)
    result = evaluate(best, root, val[:1], tmp_path/'evaluation', tile_size=128, overlap=32)
    assert result['image_count'] == 1
    assert all(np.isfinite(result['means'][key]) for key in ('psnr_denoised', 'ssim_denoised', 'uqi_denoised'))
    export_model_d(output/'best.keras', tmp_path/'trained-export')


def test_d_interrupted_training_resume(tmp_path, monkeypatch):
    import csv
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
    cfg = load_config('configs/model_d.yaml')
    cfg['data'].update(root=str(root), split=str(split), expected_pairs=10, patch_size=16, patches_per_image=1)
    cfg['training'].update(output=str(tmp_path/'run'), epochs=2)
    original = Recovery.on_epoch_end
    before = {}
    def interrupt(self, epoch, logs=None):
        original(self, epoch, logs)
        before['optimizer'] = [var.numpy().copy() for var in self.model.optimizer.variables]
        raise RuntimeError('simulated interruption')
    monkeypatch.setattr(Recovery, 'on_epoch_end', interrupt)
    with pytest.raises(RuntimeError, match='simulated'):
        train_experiment(cfg)
    restored, state = load_recovery(tmp_path/'run')
    assert state['epoch'] == 1
    for actual, expected in zip(restored.optimizer.variables, before['optimizer']):
        np.testing.assert_array_equal(actual, expected)
    monkeypatch.setattr(Recovery, 'on_epoch_end', original)
    output = train_experiment(cfg, resume=True)
    model, state = load_recovery(output)
    assert state['epoch'] == 2 and int(model.optimizer.iterations) == 2
    assert split.read_bytes() == split_bytes
    assert [r['epoch'] for r in csv.DictReader((output/'training_history.csv').open())] == ['0', '1']


def test_d_requires_original_split_and_counts(tmp_path, monkeypatch):
    from src.dataset import Pair, save_split
    from src.train import train_experiment
    import src.train as train_module
    cfg = load_config('configs/model_d.yaml')
    pairs = [Pair(f'scene_{i:03}', str(j), f'scene_{i:03}/NOISY_SRGB_{j:03}.PNG',
                  f'scene_{i:03}/GT_SRGB_{j:03}.PNG') for i in range(160) for j in (10, 11)]
    monkeypatch.setattr(train_module, 'discover_pairs', lambda *args: pairs)
    cfg['data'].pop('split')
    with pytest.raises(ValueError, match='Model D requires an existing shared split'):
        train_experiment(cfg)
    split = tmp_path/'wrong.json'
    save_split(split, tmp_path, pairs[:254], pairs[254:], 42)
    original_bytes = split.read_bytes()
    cfg['data']['split'] = str(split)
    with pytest.raises(ValueError, match='Model D requires the original 256/64'):
        train_experiment(cfg)
    assert split.read_bytes() == original_bytes
