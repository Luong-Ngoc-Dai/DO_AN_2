"""Model C architecture, storage estimates, RTL tensor order and training smoke."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from src.common import load_config
from src.model import build_model, build_cae, build_skip_cae, HARDWARE_FILTERS, complexity
from src.hardware import describe_layers, memory_report
from src.export_weights import export_weights, sha256


def model_c():
    return build_model(load_config('configs/hardware_cae.yaml')['model'])


def test_model_c_config_and_graph():
    cfg = load_config('configs/hardware_cae.yaml')
    assert cfg['seed'] == 42
    assert cfg['data']['split'] == 'runs/sidd_split.json'
    assert cfg['data']['expected_pairs'] == 320
    assert cfg['data']['patch_size'] == 128 and cfg['data']['patches_per_image'] == 4
    assert cfg['training']['batch_size'] == 8 and cfg['training']['epochs'] == 100
    assert cfg['training']['learning_rate'] == .0001
    model = model_c()
    convs = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    assert len(convs) == 10
    assert [layer.filters for layer in convs] == HARDWARE_FILTERS
    assert [layer.strides for layer in convs] == [(1, 1), (2, 2), (1, 1), (2, 2)]+[(1, 1)]*6
    assert all(layer.kernel_size == (3, 3) and layer.padding == 'same' for layer in convs)
    assert [layer.activation.__name__ for layer in convs] == ['relu']*9+['linear']
    assert sum(isinstance(layer, tf.keras.layers.Add) for layer in model.layers) == 1
    assert all(isinstance(layer, (tf.keras.layers.InputLayer, tf.keras.layers.Conv2D,
                                 tf.keras.layers.UpSampling2D, tf.keras.layers.Add)) for layer in model.layers)
    upsamples = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.UpSampling2D)]
    assert len(upsamples) == 2 and all(layer.size == (2, 2) and layer.interpolation == 'nearest' for layer in upsamples)
    assert model.get_config() == build_cae(HARDWARE_FILTERS).get_config()
    with pytest.raises(ValueError, match='fixed Model C'):
        build_model({'architecture': 'hardware_cae', 'filters': [32]*9+[3]})


@pytest.mark.parametrize('shape', [(128, 128), (128, 256), (256, 256)])
def test_model_c_forward(shape):
    model = model_c()
    x = tf.random.uniform((1, *shape, 3))
    y = model(x)
    assert y.shape == x.shape and bool(tf.reduce_all(tf.math.is_finite(y)))
    final = model.get_layer('conv_10')
    final.set_weights([np.zeros_like(weight) for weight in final.get_weights()])
    np.testing.assert_array_equal(model(x), x)


def test_model_c_costs_and_padding():
    model = model_c()
    info = complexity(model)
    assert info['parameters'] == 61011
    assert info['conv_macs'] == 273678336 and info['conv_flops'] == 547356672
    # Independent convolution arithmetic from specified input sizes and channels.
    sizes = [128, 64, 64, 32, 32, 64, 64, 128, 128, 128]
    channels = [3]+HARDWARE_FILTERS[:-1]
    assert sum(9*ci*co+co for ci, co in zip(channels, HARDWARE_FILTERS)) == 61011
    assert sum(hw*hw*9*ci*co for hw, ci, co in zip(sizes, channels, HARDWARE_FILTERS)) == info['conv_macs']
    layers = describe_layers(model)
    memory = memory_report(model, layers)
    assert memory['conv2d_count'] == 10
    assert memory['parameter_storage_bytes'] == {'FP32': 244044, 'INT16': 122022, 'INT8': 61011}
    assert memory['kernel_elements']+memory['bias_elements'] == model.count_params()
    assert memory['largest_feature_map']['shape'] == [128, 128, 32]
    assert memory['largest_feature_map']['storage_bytes']['FP32'] == 2097152
    assert memory['largest_conv_feature_map']['shape'] == [128, 128, 16]
    assert memory['intermediate_memory']['peak_live_storage_bytes']['FP32'] == 3342336
    for row in layers:
        if row['operation'] == 'Conv2D':
            assert row['padding_top_bottom'] == ([0, 1] if row['strides'] == [2, 2] else [1, 1])
            assert row['padding_left_right'] == row['padding_top_bottom']
    with pytest.raises(ValueError, match='positive'):
        describe_layers(model, 0, 128)


def numpy_conv(x, kernel, bias, strides, relu):
    """Independent NHWC/HWIO SAME cross-correlation for small RTL reference inputs."""
    h, w = x.shape[:2]
    sh, sw = strides
    oh, ow = (h+sh-1)//sh, (w+sw-1)//sw
    ph, pw = max((oh-1)*sh+3-h, 0), max((ow-1)*sw+3-w, 0)
    padded = np.pad(x, ((ph//2, ph-ph//2), (pw//2, pw-pw//2), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(0, 1))[::sh, ::sw]
    y = np.einsum('xycij,ijco->xyo', windows, kernel)+bias
    return np.maximum(y, 0) if relu else y


def test_export_roundtrip_and_numpy_reference(tmp_path):
    tf.keras.utils.set_random_seed(123)
    model = model_c()
    checkpoint = tmp_path/'best.keras'
    model.save(checkpoint)
    original_hash = sha256(checkpoint)
    output = export_weights(checkpoint, tmp_path/'export', height=8, width=12)
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['checkpoint_sha256'] == original_hash == sha256(checkpoint)
    assert len(manifest['layers']) == 10 and manifest['flatten_order'] == 'C'
    convs = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    for layer, entry in zip(convs, manifest['layers']):
        assert layer.name == entry['name']
        for array, tensor in zip(layer.get_weights(), entry['tensors'].values()):
            npy = np.load(output/tensor['npy'], allow_pickle=False)
            csv_tensor = np.loadtxt(output/tensor['csv'], skiprows=1, dtype=np.float32).reshape(array.shape)
            np.testing.assert_array_equal(npy, array)
            np.testing.assert_array_equal(csv_tensor, array)
            assert sha256(output/tensor['npy']) == tensor['npy_sha256']
    restored = model_c()
    for layer, entry in zip([l for l in restored.layers if isinstance(l, tf.keras.layers.Conv2D)], manifest['layers']):
        layer.set_weights([np.load(output/entry['tensors'][key]['npy']) for key in ('kernel', 'bias')])
    noisy = np.random.default_rng(1).random((8, 12, 3), dtype=np.float32)
    np.testing.assert_array_equal(model(noisy[None]), restored(noisy[None]))
    # Compute every convolution/upsample/Add using exported values independently
    # of Keras; check both internal correction and full RGB output.
    x = noisy.copy()
    exported = {entry['name']: entry for entry in manifest['layers']}
    reference = tf.keras.Model(model.input, [layer.output for layer in model.layers])(noisy[None])
    for layer, expected in zip(model.layers, reference):
        if isinstance(layer, tf.keras.layers.Conv2D):
            entry = exported[layer.name]
            kernel, bias = [np.load(output/entry['tensors'][key]['npy']) for key in ('kernel', 'bias')]
            x = numpy_conv(x, kernel, bias, layer.strides, layer.activation.__name__ == 'relu')
        elif isinstance(layer, tf.keras.layers.UpSampling2D):
            x = np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)
        elif isinstance(layer, tf.keras.layers.Add):
            x = noisy+x
        np.testing.assert_allclose(x, expected.numpy()[0], atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match='already exists'):
        export_weights(checkpoint, output)


@pytest.mark.parametrize('builder', [build_cae, lambda: build_skip_cae(HARDWARE_FILTERS)])
def test_export_rejects_other_graphs(tmp_path, builder):
    path = tmp_path/'other.keras'
    builder().save(path)
    with pytest.raises(ValueError, match='fixed Model C graph'):
        export_weights(path, tmp_path/'export')
    assert not (tmp_path/'export').exists()


def test_model_c_smoke_and_export(tmp_path):
    from test_pipeline import fake_sidd
    from src.dataset import discover_pairs, split_pairs, save_split
    from src.train import train_experiment
    from src.checkpoint import load_recovery
    root = tmp_path/'data'
    fake_sidd(root)
    train, val = split_pairs(discover_pairs(root), seed=42)
    shared = tmp_path/'sidd_split.json'
    save_split(shared, root, train, val, 42)
    before = shared.read_bytes()
    cfg = load_config('configs/hardware_cae.yaml')
    cfg['data'].update(root=str(root), split=str(shared), expected_pairs=10)
    cfg['training']['output'] = str(tmp_path/'run')
    output = train_experiment(cfg, smoke=True)
    assert shared.read_bytes() == before
    assert json.loads((output/'training_summary.json').read_text())['epochs_executed'] == 1
    for name in ('best.keras', 'restored_best.keras', 'quality_history.png', 'model_summary.txt'):
        assert (output/name).is_file()
    records = list(csv.DictReader((output/'training_history.csv').open()))
    assert np.isfinite(float(records[0]['val_psnr'])) and np.isfinite(float(records[0]['val_ssim']))
    recovered, state = load_recovery(output)
    assert state['epoch'] == 1 and int(recovered.optimizer.iterations) == 1
    export_weights(output/'best.keras', tmp_path/'trained-export')


def test_model_c_interrupt_and_resume(tmp_path, monkeypatch):
    from test_pipeline import fake_sidd
    from src.dataset import discover_pairs, split_pairs, save_split
    from src.train import train_experiment
    from src.checkpoint import Recovery, load_recovery
    root = tmp_path/'data'
    fake_sidd(root)
    train, val = split_pairs(discover_pairs(root), seed=42)
    shared = tmp_path/'sidd_split.json'
    save_split(shared, root, train, val, 42)
    split_bytes = shared.read_bytes()
    cfg = load_config('configs/hardware_cae.yaml')
    # Small synthetic crops to validate interruption mechanics, not SIDD quality.
    cfg['data'].update(root=str(root), split=str(shared), expected_pairs=10, patch_size=16, patches_per_image=1)
    cfg['training'].update(output=str(tmp_path/'run'), epochs=2)
    original = Recovery.on_epoch_end
    def interrupt(self, epoch, logs=None):
        original(self, epoch, logs)
        raise RuntimeError('simulated interruption')
    monkeypatch.setattr(Recovery, 'on_epoch_end', interrupt)
    with pytest.raises(RuntimeError, match='simulated'):
        train_experiment(cfg)
    monkeypatch.setattr(Recovery, 'on_epoch_end', original)
    output = train_experiment(cfg, resume=True)
    recovered, state = load_recovery(output)
    assert state['epoch'] == 2 and int(recovered.optimizer.iterations) == 2
    assert recovered.count_params() == 61011
    assert [r['epoch'] for r in csv.DictReader((output/'training_history.csv').open())] == ['0', '1']
    assert shared.read_bytes() == split_bytes
    assert json.loads((output/'training_summary.json').read_text())['epochs_executed'] == 2


def test_model_c_requires_unchanged_shared_split(tmp_path, monkeypatch):
    from src.dataset import Pair, save_split
    from src.train import train_experiment
    import src.train as train_module
    cfg = load_config('configs/hardware_cae.yaml')
    pairs = [Pair(f'scene_{i:03}', str(j), f'scene_{i:03}/NOISY_SRGB_{j:03}.PNG',
                  f'scene_{i:03}/GT_SRGB_{j:03}.PNG') for i in range(160) for j in (10, 11)]
    monkeypatch.setattr(train_module, 'discover_pairs', lambda *args: pairs)
    cfg['data'].pop('split')
    with pytest.raises(ValueError, match='existing shared split'):
        train_experiment(cfg)
    # Wrong scene-disjoint membership counts must fail before opening images.
    split = tmp_path/'wrong_split.json'
    save_split(split, tmp_path, pairs[:254], pairs[254:], 42)
    cfg['data']['split'] = str(split)
    original = split.read_bytes()
    with pytest.raises(ValueError, match='256/64'):
        train_experiment(cfg)
    assert split.read_bytes() == original
