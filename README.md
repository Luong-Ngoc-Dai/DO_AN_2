# Đồ án 2 — CAE khử nhiễu SIDD

Đọc **`src/model.py` trước**: từng lớp được khai báo trực tiếp, chia rõ Encoder → Bottleneck → Decoder → global residual. Đây là CAE baseline riêng, không tái lập PSL-AE, không pretrained weights.

| Thành phần | Kiến trúc mặc định |
|---|---|
| Encoder | Conv 32/s1 → 32/s2 → 64/s1 → 64/s2, ReLU |
| Bottleneck | Conv 96/s1, ReLU |
| Decoder | Upsample ×2 → Conv 64 → 64 → Upsample ×2 → Conv 32 → 32 → 3 |
| Output | Conv cuối linear; cộng với input noisy |

Tất cả Conv kernel 3×3, `same`; upsampling nearest. **10 Conv2D, 241.827 parameters**. Filters đọc từ `configs/baseline.yaml`. MSE train dùng đầu ra chưa clip để giữ gradient; validation/evaluate/inference clip về [0,1].

Các file cần đọc:

- `src/model.py`: kiến trúc, `model.summary()` và MACs/FLOPs.
- `src/dataset.py`: ghép NOISY/GT, chia scene, paired crops 128×128; không resize toàn ảnh.
- `src/train.py`: train/validation, checkpoint theo val_psnr, early stopping, giảm LR và history.
- `src/evaluate.py`: báo cáo PSNR/SSIM/UQI từng ảnh và improvement.
- `src/inference.py`: overlap tiled inference ảnh lớn, lưu ảnh và hình so sánh.

Giữ `metrics.py` để train/evaluate/inference dùng cùng công thức; `common.py` để dùng chung config, seed, GPU và xử lý lỗi. `compare.py` là tiện ích tùy chọn. Các wrapper ở root vẫn chạy được; ưu tiên `python -m src.<module>`.

## Cài đặt Windows / VS Code

Cài Python 3.12 64-bit, VS Code và extension Python. Mở thư mục project, chạy PowerShell tại root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -rs
.\.venv\Scripts\python.exe -m src.model
```

Chọn interpreter `.venv\Scripts\python.exe`. Summary có H/W `None` vì input linh hoạt; phần MACs tính cho 128×128: 1.066.401.792 MACs, 2.132.803.584 FLOPs (1 MAC = 2 FLOPs, chỉ convolution). Native Windows TensorFlow 2.20 chạy CPU; GPU NVIDIA cần WSL2 phù hợp. Không có GPU thì chương trình báo rõ.

## Chạy theo thứ tự

**1. Kiểm tra dataset và lưu split**

```powershell
$siddRoot = 'C:\Nam_4_HK1\Do_an_2\dataset\SIDD_Medium_Srgb\mnt\d\SIDD_Medium_Srgb\Data'
.\.venv\Scripts\python.exe -m src.dataset --data "$siddRoot" --expected-pairs 320 --seed 42 --output runs\sidd_split.json
```

Hỗ trợ `0006_NOISY_SRGB_010.PNG`/`0006_GT_SRGB_010.PNG` và tên không có tiền tố. Ghép theo thư mục scene + prefix + index. Chia 80/20 **scene trước crop**, lưu seed và đường dẫn; ảnh cùng scene không lọt vào hai tập. Validation center crop cố định, không cập nhật weights.

**2. Train thử vài batch thật**

```powershell
.\.venv\Scripts\python.exe -m src.train --data "$siddRoot" --split runs\sidd_split.json --output runs\smoke_01 --smoke
```

Smoke chỉ 1 epoch, tối đa 2 batches train và 2 ảnh validation; kết quả nằm trong `runs/smoke_01/smoke/`. Mỗi lần chạy chọn output mới.

**3. Train chính thức khi smoke đã chạy thành công**

```powershell
.\.venv\Scripts\python.exe -m src.train --data "$siddRoot" --split runs\sidd_split.json --output runs\baseline_01
```

Mặc định Adam 1e-4, MSE, batch 8, tối đa 100 epochs. Sửa YAML để đổi filters/batch/epochs. Lưu `best.keras`, `training_history.csv`, `quality_history.png`, cấu hình và split của thí nghiệm.

**4. Evaluate**

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --model runs\baseline_01\best.keras --split runs\baseline_01\split.json --data "$siddRoot" --output runs\baseline_01\evaluation
```

Lưu `metrics_per_image.csv`, `summary.json`; trung bình metric theo từng ảnh RGB [0,1]. Đây là **validation đã dùng chọn model, không phải test độc lập**. Ảnh giống nhau: PSNR +∞, SSIM/UQI 1.

**5. Inference**

```powershell
$sceneDir = Join-Path $siddRoot '0006_001_S6_00100_00060_4400_H'
.\.venv\Scripts\python.exe -m src.inference --model runs\baseline_01\best.keras --input "$sceneDir\0006_NOISY_SRGB_010.PNG" --gt "$sceneDir\0006_GT_SRGB_010.PNG" --tile-size 256 --overlap 64 --output runs\demo_01
```

Bỏ `--gt` nếu không có GT. Lưu NOISY/DENOISED/GT và hình so sánh; tiled inference giữ kích thước gốc. RAM/VRAM thiếu thì giảm batch/tile; inference vẫn cần RAM cho ảnh và buffers toàn ảnh.

## Kiểm thử và giới hạn

Tests kiểm tra paired crop, chia scene, metrics, checkpoint, inference kích thước lẻ; test refactor so output 128/256 với cấu trúc cũ **cùng weights**, và kiểm tra metric không đổi. Test 320 cặp giả không xác nhận SIDD thật. Để kiểm tra dữ liệu thật trên Windows:

```powershell
$env:SIDD_DATA_ROOT = $siddRoot
.\.venv\Scripts\python.exe -m pytest tests/test_dataset_filenames.py::test_real_sidd_320_pairs -q -rs
```

Cloud hiện không có SIDD thật/GPU: chỉ kiểm thử synthetic trên CPU, chưa xác nhận training SIDD. Hướng dẫn chi tiết về dữ liệu, protocol, so sánh model và tài nguyên: [docs/DETAILS.md](docs/DETAILS.md).
