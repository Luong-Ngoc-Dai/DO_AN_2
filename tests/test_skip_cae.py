"""Model B graph, legacy checkpoints and committed training recovery."""
import json
import numpy as np
import pytest
import tensorflow as tf
from src.model import build_cae, build_skip_cae, build_model, complexity
from src.checkpoint import Recovery, load_recovery
from src.common import load_config


@pytest.mark.parametrize('size', [128, 256])
def test_skip_shapes_and_output(size):
    model = build_skip_cae()
    assert model.count_params() == build_cae().count_params() == 241827
    convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    assert len(convs) == 10
    assert all(l.kernel_size == (3, 3) for l in convs)
    assert not any(isinstance(l, tf.keras.layers.Concatenate) for l in model.layers)
    x = tf.random.uniform((1, size, size, 3))
    assert model(x).shape == x.shape
    assert bool(tf.reduce_all(tf.math.is_finite(model(x))))
    for name, resolution, channels, source in [('skip_half', size//2, 64, 'conv_3'),
                                               ('skip_full', size, 32, 'conv_1')]:
        layer = model.get_layer(name)
        assert layer.input[1]._keras_history.operation.name == source
        tensors = tf.keras.Model(model.input, [*layer.input, layer.output])(x)
        assert all(t.shape == (1, resolution, resolution, channels) for t in tensors)
        np.testing.assert_array_equal(tensors[2], tensors[0]+tensors[1])
    model.get_layer('conv_10').set_weights([np.zeros_like(w) for w in model.get_layer('conv_10').get_weights()])
    np.testing.assert_array_equal(model(x), x)
    assert complexity(model) == complexity(build_cae())


def test_selection_and_invalid_filters():
    assert build_model({}).name == 'residual_lightweight_cae'
    assert build_model(load_config('configs/skip_cae.yaml')['model']).name == 'skip_residual_cae'
    with pytest.raises(ValueError, match='Unknown'):
        build_model({'architecture': 'typo'})
    with pytest.raises(ValueError, match='H/2'):
        build_skip_cae([32, 32, 64, 64, 96, 64, 32, 32, 32, 3])
    with pytest.raises(ValueError, match='H skip'):
        build_skip_cae([16, 32, 64, 64, 96, 64, 64, 32, 32, 3])
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        build_skip_cae()(tf.zeros((1, 127, 129, 3)))


@pytest.mark.parametrize('builder', [build_cae, build_skip_cae])
def test_full_checkpoint_and_optimizer_recovery(tmp_path, builder):
    model = builder([4, 4, 8, 8, 12, 8, 8, 4, 4, 3])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
    x = tf.random.uniform((1, 16, 16, 3))
    model.train_on_batch(x, x*.9)
    model.save(tmp_path/'best.keras')
    loaded = tf.keras.models.load_model(tmp_path/'best.keras')
    np.testing.assert_array_equal(model(x), loaded(x))
    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(tmp_path/'best.keras'), monitor='val_psnr', mode='max', save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor='val_psnr', mode='max', patience=3),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_psnr', mode='max', patience=2)]
    for callback in callbacks:
        callback.set_model(model)
        callback.on_train_begin()
        callback.best, callback.wait = 23., 1
    (tmp_path/'training_history.csv').write_text('epoch,loss,val_psnr\n0,0.1,23\n')
    recovery = Recovery(tmp_path, callbacks)
    recovery.set_model(model)
    recovery.on_epoch_end(0)
    # An interrupted uncommitted generation and stray CSV row must be ignored.
    (tmp_path/'recovery'/'epoch-incomplete').mkdir()
    with (tmp_path/'training_history.csv').open('a') as f:
        f.write('1,999,999\n')
    restored, state = load_recovery(tmp_path)
    assert state['epoch'] == 1
    assert '999' not in (tmp_path/'training_history.csv').read_text()
    np.testing.assert_array_equal(model(x), restored(x))
    for a, b in zip(model.optimizer.variables, restored.optimizer.variables):
        np.testing.assert_array_equal(a, b)
    for callback in callbacks:
        callback.on_train_begin()
    restore_callback = Recovery(tmp_path, callbacks, state)
    restore_callback.on_train_begin()
    assert all(c.best == 23 and c.wait == 1 for c in callbacks)
    model.train_on_batch(x, x*.9)
    restored.train_on_batch(x, x*.9)
    for a, b in zip(model.weights, restored.weights):
        np.testing.assert_array_equal(a, b)


def test_old_full_keras_checkpoint(tmp_path):
    original = tf.keras.models.model_from_json(open('tests/fixtures/cae_before_refactor.json').read())
    original.save(tmp_path/'best.keras')
    loaded = tf.keras.models.load_model(tmp_path/'best.keras', compile=False)
    baseline = build_model({'architecture': 'baseline'})
    baseline.set_weights(loaded.get_weights())
    x = tf.random.uniform((1, 32, 32, 3))
    np.testing.assert_array_equal(original(x), loaded(x))
    np.testing.assert_array_equal(original(x), baseline(x))


def test_interrupted_pipeline_resume(tmp_path, monkeypatch):
    from test_pipeline import fake_sidd
    from src.train import train_experiment
    root = tmp_path/'data'
    fake_sidd(root)
    cfg = load_config('configs/skip_cae.yaml')
    from src.dataset import discover_pairs, split_pairs, save_split
    train, val = split_pairs(discover_pairs(root), seed=42)
    shared_split = tmp_path/'sidd_split.json'
    save_split(shared_split, root, train, val, 42)
    split_bytes = shared_split.read_bytes()
    cfg['data']['split'] = str(shared_split)
    cfg['data'].update(root=str(root), expected_pairs=10, patch_size=16, patches_per_image=1)
    cfg['model']['filters'] = [4, 4, 8, 8, 12, 8, 8, 4, 4, 3]
    cfg['training'].update(output=str(tmp_path/'run'), batch_size=8, epochs=2)
    original = Recovery.on_epoch_end
    def interrupt(self, epoch, logs=None):
        original(self, epoch, logs)
        raise RuntimeError('simulated interruption')
    monkeypatch.setattr(Recovery, 'on_epoch_end', interrupt)
    with pytest.raises(RuntimeError, match='simulated'):
        train_experiment(cfg)
    monkeypatch.setattr(Recovery, 'on_epoch_end', original)
    output = train_experiment(cfg, resume=True)
    summary = json.loads((output/'training_summary.json').read_text())
    assert summary['epochs_executed'] == 2
    import csv
    rows = list(csv.DictReader((output/'training_history.csv').open()))
    assert [r['epoch'] for r in rows] == ['0', '1']
    model, state = load_recovery(output)
    assert state['epoch'] == 2
    assert shared_split.read_bytes() == split_bytes
    assert int(model.optimizer.iterations) == 2
    changed = json.loads(json.dumps(cfg))
    changed['model']['architecture'] = 'baseline'
    with pytest.raises(ValueError, match='Resume config'):
        train_experiment(changed, resume=True)


def test_architecture_report(tmp_path):
    from src.architecture_report import report
    rows = report(tmp_path, 16, 16)
    assert rows['baseline']['parameters'] == rows['skip_cae']['parameters']
    assert rows['baseline']['conv_macs'] == rows['skip_cae']['conv_macs']
    assert rows['skip_cae']['scalar_adds'] - rows['baseline']['scalar_adds'] == 8*8*64+16*16*32
    assert (tmp_path/'skip_cae_summary.txt').is_file()
    assert json.loads((tmp_path/'comparison.json').read_text()) == rows
