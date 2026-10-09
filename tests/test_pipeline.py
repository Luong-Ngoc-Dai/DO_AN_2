from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
import tensorflow as tf
from src.dataset import (discover_pairs, split_pairs, save_split, load_split, make_dataset,
                         read_crop, pair_size, validate_pairs)
from src.model import build_cae, complexity
from src.metrics import image_metrics
from src.inference import tiled_denoise, denoise_array, comparison_report
from src.evaluate import evaluate
from src.common import load_config
from src.train import train_experiment, ValidationQuality


def fake_sidd(root, identical=False):
    rng = np.random.default_rng(4)
    for i in range(5):
        scene = root/f'{i:04d}_scene'
        scene.mkdir(parents=True)
        for j in (10,11):
            gt = rng.integers(30,220,(140,144,3),dtype=np.uint8)
            noisy = gt if identical else np.clip(gt.astype(float)+rng.normal(0,10,gt.shape),0,255).astype(np.uint8)
            Image.fromarray(gt).save(scene/f'GT_SRGB_{j:03}.PNG')
            Image.fromarray(noisy).save(scene/f'NOISY_SRGB_{j:03}.PNG')


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path/'data'
    fake_sidd(root)
    return root


def test_pairs_split_reproducibility(dataset,tmp_path):
    pairs = discover_pairs(dataset,10)
    train,val = split_pairs(pairs,.2,42)
    assert len(train)==8 and len(val)==2
    assert not {p.scene for p in train}&{p.scene for p in val}
    assert split_pairs(pairs,.2,42)==(train,val)
    for pair in pairs:
        assert Path(pair.noisy).name.replace('NOISY','GT')==Path(pair.gt).name
    save_split(tmp_path/'split.json',dataset,train,val,42)
    root,t,v,seed = load_split(tmp_path/'split.json')
    assert root==dataset and t==train and v==val and seed==42
    with pytest.raises(ValueError,match='Expected 320'):
        discover_pairs(dataset,320)


def test_aligned_crop_and_augmentation(tmp_path):
    root=tmp_path/'identical'
    fake_sidd(root,identical=True)
    pairs=discover_pairs(root)
    x,y=read_crop(root,pairs[0],(3,4,131,132))
    np.testing.assert_array_equal(x,y)
    with Image.open(root/pairs[0].gt) as im:
        np.testing.assert_array_equal(x,np.asarray(im,dtype=np.float32)[4:132,3:131]/255)
    ds=make_dataset(root,pairs,128,2,True,42,1)
    x,y=next(iter(ds))
    np.testing.assert_array_equal(x,y)
    replay=make_dataset(root,pairs,128,2,True,42,1)
    np.testing.assert_array_equal(x,next(iter(replay))[0])
    vd=make_dataset(root,pairs,128,2,False)
    np.testing.assert_array_equal(next(iter(vd))[0],next(iter(vd))[0])


def test_errors(dataset):
    pair=discover_pairs(dataset)[0]
    with pytest.raises(MemoryError,match='budget'):
        read_crop(dataset,pair,(0,0,128,128),.001)
    with pytest.raises(ValueError,match='divisible'):
        validate_pairs(dataset,[pair],127)
    (dataset/pair.gt).unlink()
    with pytest.raises(ValueError,match='Missing'):
        discover_pairs(dataset)
    with pytest.raises(ValueError,match='does not exist'):
        discover_pairs(dataset/'missing')


def test_metrics():
    rng=np.random.default_rng(2)
    a=rng.uniform(.1,.8,(16,16,3))
    m=image_metrics(a,a)
    assert np.isinf(m['PSNR']) and m['SSIM']==pytest.approx(1) and m['UQI']==pytest.approx(1)
    assert image_metrics(a,a+.1)['PSNR']==pytest.approx(20)
    assert image_metrics(np.zeros_like(a),np.zeros_like(a))['UQI']==pytest.approx(1)
    with pytest.raises(ValueError):
        image_metrics(a,a+2)


def test_model_forward_residual_and_save(tmp_path):
    tf.keras.utils.set_random_seed(42)
    model=build_cae()
    convs=[l for l in model.layers if isinstance(l,tf.keras.layers.Conv2D)]
    assert len(convs)==10 and model.count_params()==241827
    assert model.get_layer('bottleneck').filters==96
    assert convs[-1].activation.__name__=='linear'
    assert not any(isinstance(l,(tf.keras.layers.Dense,tf.keras.layers.Flatten)) for l in model.layers)
    for size in (128,256):
        x=tf.random.uniform((1,size,size,3))
        assert model(x).shape==x.shape
    for layer in convs:
        layer.set_weights([np.zeros_like(w) for w in layer.get_weights()])
    np.testing.assert_array_equal(model(x).numpy(),x.numpy())
    assert complexity(model)['conv_flops']==2*complexity(model)['conv_macs']
    model.save(tmp_path/'test.keras')
    restored=tf.keras.models.load_model(tmp_path/'test.keras',compile=False)
    np.testing.assert_array_equal(restored(x).numpy(),x.numpy())


def test_tiling_identity_and_dimensions():
    inputs=tf.keras.Input((None,None,3))
    identity=tf.keras.Model(inputs,tf.keras.layers.Activation('linear')(inputs))
    rng=np.random.default_rng(3)
    for h,w in [(19,23),(137,169),(128,128),(1,1)]:
        x=rng.random((h,w,3),dtype=np.float32)
        y=tiled_denoise(identity,x,64,16)
        assert y.shape==x.shape
        np.testing.assert_allclose(x,y,atol=3e-7)
    with pytest.raises(ValueError,match='multiples'):
        tiled_denoise(identity,x,64,15)


def test_validation_does_not_update_weights(dataset):
    model=build_cae()
    before=[v.numpy().copy() for v in model.weights]
    callback=ValidationQuality(dataset,discover_pairs(dataset)[:1],128,1024)
    callback.set_model(model)
    logs={}
    callback.on_epoch_end(0,logs)
    assert set(logs)=={'val_psnr','val_ssim','val_uqi','val_clipped_mse'}
    for a,b in zip(before,model.weights):
        np.testing.assert_array_equal(a,b.numpy())


def test_synthetic_training_and_reports(dataset,tmp_path):
    # This is synthetic pipeline validation, never a claim about real SIDD quality.
    cfg=load_config('configs/baseline.yaml')
    cfg['data'].update(root=str(dataset),expected_pairs=10,patch_size=128,patches_per_image=1)
    cfg['training'].update(output=str(tmp_path/'experiment'),batch_size=2,epochs=1)
    output=train_experiment(cfg,smoke=True)
    assert (output/'best.keras').is_file()
    assert (output/'training_history.csv').is_file()
    assert (output/'quality_history.png').is_file()
    train=json.loads((output/'training_summary.json').read_text())
    assert train['epochs_executed']==1 and train['smoke']
    model=tf.keras.models.load_model(output/'best.keras',compile=False)
    pairs=discover_pairs(dataset)[:1]
    summary=evaluate(model,dataset,pairs,tmp_path/'evaluation',tile_size=128,overlap=32)
    assert summary['image_count']==1 and np.isfinite(summary['means']['psnr_denoised'])
    assert (tmp_path/'evaluation'/'metrics_per_image.csv').is_file()
    from src.dataset import read_pair
    x,gt=read_pair(dataset,pairs[0])
    y=tiled_denoise(model,x,128,32)
    comparison_report(x,y,gt,tmp_path/'evaluation',(4,5,64,64))
    assert (tmp_path/'evaluation'/'comparison.png').is_file()


def test_compare_same_split_and_reject_different(tmp_path,monkeypatch):
    import sys
    from src.compare import main
    summary={'split_label':'validation','image_count':1,'pair_list_sha256':'same',
             'protocol':'rgb','tile_size':128,'overlap':32,'means':{'psnr_denoised':20.0}}
    a,b=tmp_path/'a.json',tmp_path/'b.json'
    a.write_text(json.dumps(summary))
    b.write_text(json.dumps(summary))
    monkeypatch.setattr(sys,'argv',['compare',str(a),str(b),'--output',str(tmp_path/'comparison.csv')])
    main()
    assert (tmp_path/'comparison.csv').is_file()
    summary['pair_list_sha256']='different'
    b.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='Different evaluation'):
        main()
