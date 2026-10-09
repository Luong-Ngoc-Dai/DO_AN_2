"""Filename regressions. Synthetic fixtures are not the user's real SIDD dataset."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from src.dataset import (discover_pairs, split_pairs, save_split, load_split,
                         validate_pairs, make_dataset)


def write_pair(scene, prefix='', index='010'):
    scene.mkdir(parents=True, exist_ok=True)
    stem = f'{prefix}_' if prefix else ''
    pixels = np.full((128, 128, 3), 100, dtype=np.uint8)
    for kind in ('NOISY', 'GT'):
        Image.fromarray(pixels).save(scene/f'{stem}{kind}_SRGB_{index}.PNG')


@pytest.mark.parametrize('prefix,scene', [
    ('0157', '0157_scene'), ('0006', '0006_001_S6_00100_00060_4400_H'),
])
def test_four_prefixed_filenames(tmp_path, prefix, scene):
    # Each fixture has exactly the four names supplied by the user.
    for index in ('010', '011'):
        write_pair(tmp_path/scene, prefix, index)
    pairs = discover_pairs(tmp_path, expected_pairs=2)
    assert [(p.scene, p.prefix, p.index) for p in pairs] == [(scene, prefix, '10'), (scene, prefix, '11')]
    for pair, index in zip(pairs, ('010', '011')):
        assert Path(pair.noisy).name == f'{prefix}_NOISY_SRGB_{index}.PNG'
        assert Path(pair.gt).name == f'{prefix}_GT_SRGB_{index}.PNG'
    x, y = next(iter(make_dataset(tmp_path, pairs, 128, 2)))
    np.testing.assert_array_equal(x, y)


def test_prefix_is_part_of_pair_identity_and_not_split_identity(tmp_path):
    for scene in ('scene_a', 'scene_b'):
        for prefix in ('', '0006', '0157'):
            write_pair(tmp_path/scene, prefix)
    pairs = discover_pairs(tmp_path, 6)
    assert len({(p.scene, p.prefix, p.index) for p in pairs}) == 6
    train, val = split_pairs(pairs)
    assert len(train) == len(val) == 3
    assert not {p.scene for p in train} & {p.scene for p in val}
    save_split(tmp_path/'split.json', tmp_path, train, val, 42)
    _, saved_train, saved_val, _ = load_split(tmp_path/'split.json')
    assert saved_train == train and saved_val == val


def test_wrong_prefix_or_scene_never_pairs(tmp_path):
    scene = tmp_path/'scene'
    write_pair(scene, '0157')
    (scene/'0157_GT_SRGB_010.PNG').rename(scene/'0006_GT_SRGB_010.PNG')
    with pytest.raises(ValueError, match='Missing NOISY/GT'):
        discover_pairs(tmp_path)
    (scene/'0006_GT_SRGB_010.PNG').rename(scene/'GT_SRGB_010.PNG')
    with pytest.raises(ValueError, match='Missing NOISY/GT'):
        discover_pairs(tmp_path)
    other = tmp_path/'other'
    other.mkdir()
    (scene/'GT_SRGB_010.PNG').rename(other/'0157_GT_SRGB_010.PNG')
    with pytest.raises(ValueError, match='Missing NOISY/GT'):
        discover_pairs(tmp_path)


def test_prefix_and_index_duplicates_and_case(tmp_path):
    write_pair(tmp_path/'scene', '0157')
    noisy = tmp_path/'scene'/'0157_NOISY_SRGB_010.PNG'
    noisy.rename(noisy.with_name('0157_noisy_srgb_010.png'))
    assert len(discover_pairs(tmp_path)) == 1
    noisy = noisy.with_name('0157_noisy_srgb_010.png')
    with Image.open(noisy) as image:
        image.save(noisy.with_name('0157_NOISY_SRGB_10.PNG'))
    with pytest.raises(ValueError, match='Duplicate NOISY'):
        discover_pairs(tmp_path)


def test_split_validation_and_legacy_compatibility(tmp_path):
    for scene in ('scene_a', 'scene_b'):
        write_pair(tmp_path/scene)
    train, val = split_pairs(discover_pairs(tmp_path))
    path = tmp_path/'split.json'
    save_split(path, tmp_path, train, val, 42)
    legacy = json.loads(path.read_text())
    legacy['version'] = 1
    for pair in legacy['train'] + legacy['validation']:
        pair.pop('prefix')
    path.write_text(json.dumps(legacy))
    assert load_split(path)[1:3] == (train, val)
    legacy['validation'] = legacy['train']
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match='scene leakage'):
        load_split(path)


def test_saved_prefix_must_match_filename(tmp_path):
    for scene in ('scene_a', 'scene_b'):
        write_pair(tmp_path/scene, '0157')
    train, val = split_pairs(discover_pairs(tmp_path))
    path = tmp_path/'split.json'
    save_split(path, tmp_path, train, val, 42)
    data = json.loads(path.read_text())
    data['train'][0]['prefix'] = '0006'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='does not match filename'):
        load_split(path)


def test_synthetic_320_pairs_and_scene_split(tmp_path):
    # 160 tiny synthetic scene instances; does NOT validate real SIDD extraction.
    for scene_id in range(1, 161):
        prefix = f'{scene_id:04d}'
        for index in ('010', '011'):
            write_pair(tmp_path/f'{prefix}_scene', prefix, index)
    pairs = discover_pairs(tmp_path, expected_pairs=320)
    validate_pairs(tmp_path, pairs, 128)
    train, val = split_pairs(pairs, .2, 42)
    assert len(train) == 256 and len(val) == 64
    assert len({p.scene for p in train}) == 128
    assert len({p.scene for p in val}) == 32
    assert not {p.scene for p in train} & {p.scene for p in val}
    (tmp_path/pairs[0].gt).unlink()
    (tmp_path/pairs[0].noisy).unlink()
    with pytest.raises(ValueError, match='Expected 320 pairs, found 319'):
        discover_pairs(tmp_path, expected_pairs=320)


@pytest.mark.skipif(not os.environ.get('SIDD_DATA_ROOT'), reason='Real SIDD unavailable; set SIDD_DATA_ROOT to the extracted Data directory')
def test_real_sidd_320_pairs():
    root = Path(os.environ['SIDD_DATA_ROOT'])
    pairs = discover_pairs(root, expected_pairs=320)
    validate_pairs(root, pairs, 128)
    train, val = split_pairs(pairs, .2, 42)
    assert len(train) + len(val) == 320
    assert not {p.scene for p in train} & {p.scene for p in val}
