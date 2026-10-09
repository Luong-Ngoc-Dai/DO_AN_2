# Đồ án 2 — Residual Lightweight CAE / SIDD Medium sRGB

Python + TensorFlow/Keras. Đây là **CAE baseline mới**, không tái lập PSL-AE, không pretrained weights. Chuẩn bị cho thử nghiệm kiến trúc và so sánh model; chưa lượng tử hóa/triển khai FPGA. Không tự tải hoặc train đầy đủ SIDD.

## Cấu trúc

```text
configs/baseline.yaml    # cấu hình kiến trúc/dataset/train/tile
src/
  dataset.py            # quét PNG, ghép cặp, chia scene, paired random crop
  model.py              # CAE residual, summary, MACs/FLOPs
  train.py              # training, callback validation, checkpoint và history
  metrics.py            # PSNR, SSIM, UQI RGB [0,1]
  evaluate.py           # metrics từng ảnh + summary tập validation/held-out
  inference.py          # full/overlap tiled inference và hình so sánh
  common.py             # config, seed, GPU, JSON, lỗi tài nguyên
  compare.py            # tổng hợp summary nhiều thí nghiệm thành CSV
requirements.txt
pytest.ini
tests/test_pipeline.py
```

Các file Python ở thư mục gốc là entrypoint tương thích, trỏ tới `src/`. Ưu tiên lệnh `python -m src.<module>` từ thư mục project; không chạy trực tiếp `python src/train.py` vì relative imports.

## Dữ liệu và chia tập

Ví dụ đường dẫn bạn đã cung cấp:

```text
C:\Nam_4_HK1\Do_an_2\dataset\SIDD_Medium_Srgb\mnt\d\SIDD_Medium_Srgb\Data
  0001_001_S6_00100_00060_3200_L\
    NOISY_SRGB_010.PNG
    GT_SRGB_010.PNG
    NOISY_SRGB_011.PNG
    GT_SRGB_011.PNG
```

Quét đệ quy, khóa ghép cặp = đường dẫn tương đối của scene instance + chỉ số ảnh. Kiểm tra thiếu partner, trùng index/kind, tên sRGB bất thường, RGB 8-bit và kích thước tương ứng. Mặc định yêu cầu **320 cặp**; cấu hình số khác chỉ khi bạn chủ động dùng subset. Không đọc RAW/MAT.

Chia **80%/20% số scene instances**, trước crop, seed 42. Khi số scene không chia hết cho 5, làm tròn số scene validation; tỷ lệ số cặp có thể không đúng 80/20 nếu các scene có số cặp khác nhau. `split.json` lưu seed, scene/index và đường dẫn tương đối, dùng `--data` để đổi root khi chuyển máy. Hai cặp cùng scene luôn cùng tập. Đây là scene **instance** (thư mục), không phải gộp mọi lần chụp của cùng physical scene.

Train mặc định 4 random crops 128×128/cặp/epoch, thay đổi qua epoch, rotate 90°/flip đồng bộ NOISY/GT. Validation một center crop cố định/cặp. Không resize toàn ảnh. `tf.data` generator chỉ giữ một cặp uint8 đã decode + batch patch float32 + một batch prefetch; không cache toàn dataset. PNG vẫn phải decode toàn ảnh mỗi lần đọc, nhưng reuse cùng cặp cho nhiều crops, chỉ chuyển crop sang float32.

## Kiến trúc và residual learning

| Conv | Filters | Kernel | Stride | Activation |
|---|---:|---|---:|---|
| 1 | 32 | 3×3 | 1 | ReLU |
| 2 | 32 | 3×3 | 2 | ReLU |
| 3 | 64 | 3×3 | 1 | ReLU |
| 4 | 64 | 3×3 | 2 | ReLU |
| 5 bottleneck | 96 | 3×3 | 1 | ReLU |
| 6 | 64 | 3×3 | 1 | ReLU |
| 7 | 64 | 3×3 | 1 | ReLU |
| 8 | 32 | 3×3 | 1 | ReLU |
| 9 | 32 | 3×3 | 1 | ReLU |
| 10 | 3 | 3×3 | 1 | Linear |

Input `(None,None,3)`, all padding `same`. Nearest upsampling ×2 trước Conv 6 và 8. Không Dense/Flatten/classifier/multiple decoders. Có **241.827 parameters**. Sửa danh sách 10 filters trong YAML để thử biến thể; lớp cuối phải có 3 channels. Strides/kernel được tập trung ở `build_cae` nếu cần đổi kiến trúc sâu hơn.

Graph training trả `raw = noisy + correction`. Loss MSE so **raw với GT**. Không clip trong graph: clip có gradient bằng 0 khi raw ngoài [0,1], có thể khiến pixel dự đoán sai khó phục hồi. Validation log `val_loss` trên raw, `val_clipped_mse` và `val_psnr`/`val_ssim` trên **clip(raw,0,1)**. PSNR được suy ra từ MSE clipped từng ảnh; không diễn giải raw `val_loss` thành PSNR. Checkpoint/early stopping/LR scheduler dùng `val_psnr`, đồng nhất với miền ảnh đánh giá/inference. Tiled inference blend raw rồi clip một lần. PNG lượng tử hóa về uint8 chỉ khi lưu; metrics tính trước lượng tử hóa.

Input gọi model trực tiếp phải chia hết cho 4; inference tự pad phải/dưới rồi crop, output giữ đúng H×W gốc. Tiling giữ grid bội số 4 để phù hợp hai stride-2 stages, overlap + cửa sổ Hann dương để blend và giảm seam. Đây không đảm bảo tiled output giống full inference, đặc biệt vùng biên/receptive field; giữ tile/overlap cố định khi so sánh model.

**Input 128×128×3: 1.066.401.792 MACs / 2.132.803.584 FLOPs**. MACs/FLOPs tính khi chạy `src.model` và lưu `complexity.json`: conv output H×W×Cout×Kh×Kw×Cin; **1 MAC = 2 FLOPs** (multiply + add). Chỉ tính convolution, không bias/ReLU/upsampling/residual add/clip; không phải đo latency hay chi phí FPGA thực tế.

## Windows PowerShell / VS Code

Cài Python **3.12 64-bit**, VS Code và extension **Python (Microsoft)**. Mở thư mục project bằng VS Code. Terminal PowerShell, làm việc tại root có `configs/` và `src/`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.model --config configs\baseline.yaml
```

`Python: Select Interpreter` → `.venv\Scripts\python.exe`. Không cần activate/đổi ExecutionPolicy. Native Windows TensorFlow 2.20 chạy CPU; GPU NVIDIA cần WSL2 + TensorFlow/CUDA phù hợp. Chương trình kiểm tra GPU và bật memory growth nếu có; thông báo rõ khi CPU. GPU và Windows chưa được kiểm thử trong cloud Linux này.

### Kiểm tra SIDD trước training

```powershell
$siddRoot = 'C:\Nam_4_HK1\Do_an_2\dataset\SIDD_Medium_Srgb\mnt\d\SIDD_Medium_Srgb\Data'
.\.venv\Scripts\python.exe -m src.dataset --data "$siddRoot" --expected-pairs 320 --seed 42 --output runs\sidd_split.json
.\.venv\Scripts\python.exe -m src.train --config configs\baseline.yaml --data "$siddRoot" --split runs\sidd_split.json --output runs\real_smoke_01 --smoke
```

Smoke dùng dữ liệu **thật ở đường dẫn được truyền vào**, tối đa 2 batches train khi dataset đủ lớn và 2 ảnh validation, 1 epoch; output trong `runs/real_smoke_01/smoke/`. Không thể chạy bước này ở cloud khi dataset chỉ ở ổ C: Windows. Khi lặp lại smoke, chọn output mới vì chương trình từ chối ghi đè một experiment đã có config.

### Training đầy đủ — chỉ chạy sau khi kiểm tra thành công

Sửa config batch size/epochs/patience, hoặc tạo config riêng cho từng experiment. Mặc định Adam 1e-4, MSE, batch 8, tối đa 100 epochs; EarlyStopping 15, ReduceLROnPlateau 5, best model theo val_psnr.

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\baseline.yaml --data "$siddRoot" --split runs\sidd_split.json --output runs\baseline_01
```

Kết quả: `config.yaml`, `split.json`, `run_manifest.json`, `complexity.json`, `best.keras`, `restored_best.keras`, `history.json`, `training_history.csv`, `training_summary.json`, `quality_history.png` (PSNR/SSIM theo epoch). Callback validation dùng `training=False`, không gradient/optimizer update. `restored_best.keras` lưu weights đã khôi phục sau EarlyStopping; dùng `best.keras` để đánh giá checkpoint với trạng thái optimizer đồng nhất.

### Đánh giá trên validation ảnh đầy đủ

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --model runs\baseline_01\best.keras --split runs\baseline_01\split.json --data "$siddRoot" --tile-size 256 --overlap 64 --output runs\baseline_01\evaluation
```

Lưu `metrics_per_image.csv`, `summary.json`: PSNR/SSIM/UQI noisy và denoised, PSNR/SSIM improvement. Mỗi ảnh một giá trị, trung bình đều giữa ảnh (không gộp MSE mọi ảnh). Validation đã dùng chọn checkpoint, **không phải independent test**. `--heldout-data <dir>` chỉ dùng khi bạn có dataset khác thực sự độc lập; người dùng cần xác nhận không overlap với train. Validation center-crop theo epoch và full-image tiled evaluation là hai protocol khác nhau, không kỳ vọng điểm giống nhau.

### Inference + hình so sánh

```powershell
$sceneDir = Join-Path $siddRoot '0001_001_S6_00100_00060_3200_L'
.\.venv\Scripts\python.exe -m src.inference --model runs\baseline_01\best.keras --input "$sceneDir\NOISY_SRGB_010.PNG" --gt "$sceneDir\GT_SRGB_010.PNG" --tile-size 256 --overlap 64 --crop 500 500 128 128 --output runs\demo_01
```

Bỏ `--gt` nếu không có GT, bỏ `--crop` để dùng center crop. GT cần đúng cặp và cùng kích thước/căn chỉnh. Lưu NOISY/DENOISED/GT PNG và `comparison.png`: hàng full image, hàng crop cùng tọa độ, metric full image dưới hình. Không dùng ảnh chưa train để tuyên bố khử nhiễu hiệu quả.

### So sánh nhiều model

Đổi filters trong config, giữ cùng seed/split/metrics/tile. Train mỗi model vào thư mục riêng, evaluate trên cùng split:

```powershell
.\.venv\Scripts\python.exe -m src.compare runs\baseline_01\evaluation\summary.json runs\variant_01\evaluation\summary.json --output runs\comparison.csv
```

### Tài nguyên và lỗi

Thiếu dataset/partner, sai kích thước, split leakage, patch/tile invalid báo lỗi rõ. RAM budget kiểm tra ước tính trước decode/buffer; **không phải phép đo peak RAM**. Training giữ một pair decode, tiled inference vẫn giữ input/GT/accumulator toàn ảnh trong RAM nhưng chỉ một tile qua mạng, giảm VRAM. Metric SSIM/UQI cũng cần temporary arrays. Nếu RAM/VRAM hết, giảm batch/tile, đóng ứng dụng khác hoặc tăng RAM budget chỉ khi máy đủ RAM; vẫn có thể bị hệ điều hành kill nếu giới hạn thấp. Chi phí GPU activation không thể suy ra chỉ từ parameter count.

## Metric / tài liệu tham khảo

Tham khảo [PSL-AE evaluation.py](https://github.com/yunhaoyang234/Patch-Subspace-Learning-Autoencoder/blob/master/evaluation.py), không sao chép model: PSNR MSE, scikit-image SSIM, sewar UQI ws=8.

- PSNR = 10 log10(1/MSE), data_range=1, RGB normalized. Identical images → +∞, không hardcode điểm kết quả.
- SSIM `channel_axis=-1`, `data_range=1`, window 7, mặc định scikit-image.
- UQI `sewar.full_ref.uqi`, window 8, trung bình kênh RGB.
- Metric yêu cầu ảnh ít nhất 8×8; inference vẫn hỗ trợ ảnh nhỏ hơn nhưng không tính metric/report GT cho ảnh quá nhỏ.
- PSNR improvement từng ảnh = denoised − noisy; SSIM tương tự. +∞−+∞ không xác định → NaN; JSON lưu giá trị không hữu hạn bằng chuỗi để hợp lệ, CSV lưu inf/nan.

Không phải official SIDD benchmark scoring, không so trực tiếp điểm khác protocol/miền ảnh.

## Linux/cloud

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.model
```

Tests tạo PNG giả nhỏ trong thư mục tạm: ghép cặp, split 80/20 không leakage, alignment crop/augmentation, validation cố định/không đổi weights, model forward 128/256, residual identity, save/load, tiled identity/kích thước lẻ, metrics identical/known MSE, train smoke và báo cáo. Dữ liệu SIDD thật chỉ nằm trên Windows nên các bước kiểm tra 320 cặp và train thật vẫn cần bạn chạy bằng các lệnh phía trên. Chưa train full SIDD, chưa xác nhận chất lượng denoising hay FPGA readiness.

## Kết quả xác minh hiện tại

Cloud Linux CPU, Python 3.12 / TensorFlow 2.20: 9 kiểm thử đã qua, có forward 128×128 và 256×256, smoke train bằng PNG giả, lưu/nạp checkpoint, tiled inference, metrics và hình so sánh. pip check không có dependency lỗi. Có DeprecationWarning từ sewar/Keras, không ảnh hưởng kết quả kiểm thử. Chưa chạy trên Windows/GPU/SIDD thật.
