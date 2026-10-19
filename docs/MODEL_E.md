# Model E — Two-Skip Residual CAE

Model E là ablation trực tiếp từ D: giữ bảy Conv2D và toàn bộ training protocol,
chỉ thêm hai encoder–decoder Add. Không knowledge distillation, teacher, loss
mới, BatchNorm, Concatenate, attention hay projection convolution. Chưa có kết
quả SIDD thật chứng minh E tốt hơn D; chưa lượng tử hóa hoặc synthesis KV260.

## Graph D/E và chi phí

```
noisy → Conv1 → Conv2 → Conv3 → Upsample1 → Conv4 → Add(Conv2)
      → Conv5 → Upsample2 → Conv6 → Add(Conv1) → Conv7 → Add(noisy)
```

`src/model_e.py:build_two_skip_cae()` giữ filters `[16,32,48,32,16,16,3]`, strides
`[1,2,2,1,1,1,1]`, kernel 3×3 SAME, bias, ReLU Conv1–6 và Conv7 linear.
Upsampling nearest ×2 sau Conv3/Conv5. Không thêm activation sau hai skip.

- `e_skip_half`: `e_conv_4 + e_conv_2`, trước `e_conv_5`, H/2×W/2×32.
- `e_skip_full`: `e_conv_6 + e_conv_1`, trước `e_conv_7`, H×W×16.
- `e_residual_add`: `noisy_rgb + e_conv_7`. Không clip output khi train.

Input trực tiếp có H/W chia hết cho 4; tests bao gồm 128×128 và 128×256.
Tiled inference hiện có pad ảnh kích thước lẻ và crop lại. Khi cùng seed 42,
tests so sánh từng Conv config/weights khởi tạo D/E để kiểm tra chỉ graph Add
khác nhau. Không sửa builders, config hoặc checkpoint A/B/C/D.

Chạy báo cáo tính từ graph thực tế:

```powershell
.\.venv\Scripts\python.exe -m src.model --config configs\model_e.yaml
.\.venv\Scripts\python.exe -m src.report_model_e
```

| Chỉ tiêu, batch 1 RGB 128×128 | D | E |
|---|---:|---:|
| Conv2D | 7 | 7 |
| Trainable parameters | 40,195 | 40,195 |
| Convolution MACs | 160,432,128 | 160,432,128 |
| Convolution FLOPs | 320,864,256 | 320,864,256 |
| Encoder–decoder skip Add | 0 | 2 |
| Global residual Add | 1 | 1 |
| Scalar additions của hai skip | 0 | 393,216 |

Một MAC = hai FLOPs; convolution counts không gồm bias/ReLU/upsampling/Add.
Global residual thêm 49,152 scalar additions ở cả D/E; tổng Add của E là
442,368. Add không làm tăng trainable parameters nhưng không miễn phí phần cứng.

| Feature map phải giữ cho skip | Elements | FP32 bytes | INT8 bytes (giả định) |
|---|---:|---:|---:|
| Conv2: 64×64×32 | 131,072 | 524,288 (512 KiB) | 131,072 (128 KiB) |
| Conv1: 128×128×16 | 262,144 | 1,048,576 (1 MiB) | 262,144 (256 KiB) |
| Tổng hai maps | 393,216 | 1,572,864 (1.5 MiB) | 393,216 (384 KiB) |

Đây là dung lượng giữ feature maps theo full-frame, không phải tổng BRAM thực
tế. Report xuất riêng `skip_feature_maps`, `skip_feature_storage_bytes`, số Add,
weights/bias storage và timeline ước tính peak live tensors. INT8 chỉ là phép
ước lượng số bytes, không có calibration/quantization trong branch này. Streaming,
DDR tiling/fusion, line buffers, accumulators, DMA và alignment cần tính riêng.

Training vẫn tạo `complexity.json` theo convention convolution của D. Báo cáo
đầy đủ chi phí Add/buffers nằm trong `runs/model_e_architecture/complexity.json`
sau khi chạy `src.report_model_e`; bản đã tạo ở
[model_e_architecture/](model_e_architecture/).

## Windows VS Code / PowerShell / Python 3.13

Clone vào thư mục mới để không chạm outputs của các model cũ. Không cần copy
addon hay chạy installer. Dùng Python 3.13 64-bit và TensorFlow 2.20 theo requirements.
Trong VS Code chọn `.venv\Scripts\python.exe`; không cần activate venv bằng
PowerShell nên không phải đổi ExecutionPolicy.

```powershell
git clone --branch feat/model-e-two-skip --single-branch https://github.com/Luong-Ngoc-Dai/DO_AN_2.git DO_AN_2-model-e
cd .\DO_AN_2-model-e
py -3.13 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements.txt
& $py -m pip check
& $py -m pytest -q -rs
& $py -m src.report_model_e
```

Chỉnh đúng đường dẫn dataset và **split gốc đã dùng A/B/C/D**. Không tạo split
mới, không ghi đè file đã có. Dataset/SIDD split thật không được cung cấp trong
cloud; bản split synthetic trong tests không được dùng cho thí nghiệm SIDD.

```powershell
$siddRoot = 'C:\datasets\SIDD_Medium_Srgb\Data'
$originalSplit = 'C:\path\to\ModelD\runs\sidd_split.json'
New-Item -ItemType Directory -Force .\runs | Out-Null
if (-not (Test-Path .\runs\sidd_split.json)) {
    Copy-Item $originalSplit .\runs\sidd_split.json
}
if (-not (Test-Path $siddRoot)) { throw 'Sai duong dan SIDD Data' }
if (-not (Test-Path .\runs\sidd_split.json)) { throw 'Thieu split goc A/B/C/D' }
```

Smoke trước, chọn output mới nếu đã chạy một lần:

```powershell
& $py -m src.train --config configs\model_e.yaml --data "$siddRoot" --split runs\sidd_split.json --output runs\model_e_smoke --smoke
```

Smoke vẫn theo logic D: một epoch, một patch/cặp, tối đa hai batches train và
hai ảnh validation, outputs ở `runs/model_e_smoke/smoke/`. Đây không phải full
training protocol. Config chính giữ 320 cặp, split 256/64, seed 42, 128×128,
bốn patches/cặp, batch 8, Adam 1e-4, raw-output MSE, early stopping patience 15,
reduce-LR patience 5. Không thay loss, LR hoặc augmentation để bù việc thêm skip.

Chỉ train tối đa 100 epochs khi chủ động chạy lệnh này; early stopping có thể
dừng sớm như D:

```powershell
& $py -m src.train --config configs\model_e.yaml --data "$siddRoot" --split runs\sidd_split.json --output runs\model_e_100epochs
# Sau gián đoạn, dùng cùng config/data/split/output:
& $py -m src.train --config configs\model_e.yaml --data "$siddRoot" --split runs\sidd_split.json --output runs\model_e_100epochs --resume
```

Lưu `best.keras` theo validation PSNR, `restored_best.keras`, recovery,
`training_history.csv`, `history.json`, `training_summary.json`, `quality_history.png`,
`model_summary.txt` UTF-8 và `complexity.json`. Resume dùng epoch đã commit gần
nhất, gồm model/Adam/callback counters/history; epoch dang dở sẽ chạy lại. Cần
ít nhất một epoch đã commit; nếu chưa có, chạy lại ở output mới. Không dùng hai
writers cùng output. Không load checkpoint D như recovery của E; graph đã khác.

Validation training vẫn là center crop cố định/ảnh, clip [0,1] để tính PSNR/SSIM/UQI.
File metrics, inference, dataset và recovery không đổi. Không chọn model bằng
training loss hoặc đưa teacher vào pipeline.

## Evaluation và inference

Validation 64 cặp PNG bằng evaluator hiện có:

```powershell
& $py -m src.evaluate --model runs\model_e_100epochs\best.keras --split runs\sidd_split.json --data "$siddRoot" --output runs\model_e_100epochs\evaluation
```

**`evaluate_sidd_blocks.py` không có trong checkout này.** Cần cung cấp đúng bản
đã chạy A/B/C/D, command/flags và preprocessing của nó để xác nhận benchmark
SIDD Validation sRGB 1.280 block. Không viết evaluator mới hoặc tuyên bố PNG
validation tương đương. Model E serialize bằng các lớp Keras chuẩn nên không
cần custom objects để load; chỉ có thể xác nhận tương thích script block cụ thể
khi có script/dữ liệu thực tế. Giữ nguyên protocol/metrics khi so sánh D/E.

Inference ảnh đầy đủ, xuất `comparison.png` chứa NOISY/DENOISED/GT và zoom cùng vị trí:

```powershell
$noisy = 'C:\path\to\NOISY_SRGB_010.PNG'
$gt = 'C:\path\to\GT_SRGB_010.PNG'
& $py -m src.inference --model runs\model_e_100epochs\best.keras --input "$noisy" --gt "$gt" --output runs\model_e_100epochs\demo
```

Giữ cùng tile size/overlap của protocol cũ; defaults hiện có là 256/64. Nếu có
flags khác trong lệnh A/B/C/D thực tế, dùng lại đúng flags đó.

## Export FP32 weights/biases và ảnh hưởng W8A8/Verilog

```powershell
& $py -m src.export_model_e --model runs\model_e_100epochs\best.keras --output runs\model_e_100epochs\weights_export
```

Thư mục output phải mới. Exporter xác thực toàn graph E (không chỉ đếm params),
kiểm tra FP32 hữu hạn, chỉ đọc checkpoint, xuất bảy kernel/bias `.npy` và manifest.
Manifest có shapes, layer order, HWIO, C-order flatten, hashes checkpoint/tensors,
`e_skip_half`/`e_skip_full` với nguồn, consumers và shapes tại input tham chiếu 128.
Global residual được ghi riêng. TensorFlow cross-correlation không lật kernel.

W8A8/Verilog về sau cần giữ hai feature maps đến điểm Add, thêm điều khiển và
băng thông buffer; hai nhánh Add phải có scale tương thích, cộng ở độ rộng đủ rồi
requantize/saturate. Conv7 correction có dấu, không được thêm ReLU/ép unsigned.
Skip có thể làm activation range tăng, nên phải đo calibration/QAT và delta PSNR.
Không có W8A8, `.mem`, RTL hay tuyên bố board fit trong thay đổi này.

Khả năng giữ chi tiết tốt hơn chỉ là giả thuyết. Xem
[MODEL_E_VALIDATION.md](MODEL_E_VALIDATION.md) để biết các checks thực sự đã chạy;
không dùng PSNR synthetic để kết luận chất lượng trên SIDD thật.
