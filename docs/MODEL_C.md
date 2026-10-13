# Model C — Hardware-Friendly Lightweight CAE

Model C nghiên cứu sự đánh đổi chất lượng khử nhiễu SIDD và tài nguyên cho RTL
Verilog trên Kria KV260. Chưa synthesis/place-and-route; chưa kết luận model vừa
board, đạt timing hay có PSNR tốt hơn A/B.

## Kiến trúc và cấu hình

`configs/hardware_cae.yaml` chọn `model.architecture: hardware_cae`, gọi lại
`build_cae()` với `[16, 16, 32, 32, 48, 32, 32, 16, 16, 3]`. Không viết thêm mạng
trùng lặp. Tên serialized model vẫn là `residual_lightweight_cae`; phân biệt thí
nghiệm bằng YAML, filters và thư mục output, không chỉ bằng tên model.
Model A/B và defaults của `build_cae()` giữ nguyên; checkpoint cũ không cần đổi.
C không có encoder/decoder skip, BN, attention hoặc Transformer.

Tất cả Conv là kernel 3×3 SAME, có bias, TensorFlow cross-correlation, không lật
kernel. Batch-one shapes dưới đây là H×W×C, không bao gồm batch.

| Thứ tự | Lớp | Input | Output | Stride/scale | Padding top,bottom / left,right | Activation |
|---|---|---|---|---|---|---|
| 1 | conv_1 | 128×128×3 | 128×128×16 | 1 | 1,1 / 1,1 | ReLU |
| 2 | conv_2 | 128×128×16 | 64×64×16 | 2 | 0,1 / 0,1 | ReLU |
| 3 | conv_3 | 64×64×16 | 64×64×32 | 1 | 1,1 / 1,1 | ReLU |
| 4 | conv_4 | 64×64×32 | 32×32×32 | 2 | 0,1 / 0,1 | ReLU |
| 5 | bottleneck | 32×32×32 | 32×32×48 | 1 | 1,1 / 1,1 | ReLU |
| — | upsample_6 | 32×32×48 | 64×64×48 | ×2 nearest | — | — |
| 6 | conv_6 | 64×64×48 | 64×64×32 | 1 | 1,1 / 1,1 | ReLU |
| 7 | conv_7 | 64×64×32 | 64×64×32 | 1 | 1,1 / 1,1 | ReLU |
| — | upsample_8 | 64×64×32 | 128×128×32 | ×2 nearest | — | — |
| 8 | conv_8 | 128×128×32 | 128×128×16 | 1 | 1,1 / 1,1 | ReLU |
| 9 | conv_9 | 128×128×16 | 128×128×16 | 1 | 1,1 / 1,1 | ReLU |
| 10 | conv_10 | 128×128×16 | 128×128×3 | 1 | 1,1 / 1,1 | Linear |
| — | residual_reconstruction | noisy + correction, cùng 128×128×3 | 128×128×3 | Add | — | — |

SAME stride=2 trên kích thước chẵn có padding **bất đối xứng**; RTL không được
mặc định pad 1 pixel mỗi phía. Nearest ×2 lặp mỗi pixel thành khối 2×2
(`out[y,x,c] = in[y//2,x//2,c]`). Conv cuối linear giữ hiệu chỉnh âm/dương;
không clip khi train. Inference/evaluation clip RGB về [0,1] như A/B.
Input trực tiếp cần H/W chia hết cho 4; tiled inference hỗ trợ padding ảnh lẻ.

## Báo cáo A/B/C tính bằng chương trình

Chạy `python -m src.architecture_report --output runs/architecture_comparison`.
Bản xuất 128×128 đã lưu ở [architecture_comparison/comparison.json](architecture_comparison/comparison.json),
summary TXT và `<architecture>_layers.json` (input/output, channels, stride,
padding, activation, nguồn tensor). Báo cáo tính trên graph, không train.

| Model | Conv2D | Parameters | Conv MACs | Conv FLOPs |
|---|---:|---:|---:|---:|
| A | 10 | 241,827 | 1,066,401,792 | 2,132,803,584 |
| B | 10 | 241,827 | 1,066,401,792 | 2,132,803,584 |
| C | 10 | 61,011 | 273,678,336 | 547,356,672 |

1 MAC = 2 FLOPs; bảng chỉ tính convolution, không tính bias/ReLU/upsample/clipping.
Add được báo cáo riêng: A/C 49,152 scalar additions, B 835,584. C giảm khoảng
74.8% parameters và 74.3% Conv MACs so với A/B; chất lượng phải đo sau training.

| Model | Parameters FP32 (bytes) | INT16 (bytes, giả định) | INT8 (bytes, giả định) |
|---|---:|---:|---:|
| A/B | 967,308 | 483,654 | 241,827 |
| C | 244,044 | 122,022 | 61,011 |

Bảng gồm kernels **và biases**; JSON tách `weight_storage_bytes`,
`bias_storage_bytes`, `parameter_storage_bytes`. INT16/INT8 chỉ là dung lượng
nếu mọi phần tử có cùng độ rộng, không phải model đã quantize. Thiết kế thật có
thể dùng biases/accumulators rộng hơn, scale tables và padding/alignment khác.

| Model | Feature map lớn nhất (kể cả upsample) | FP32 bytes | Peak live tensor bytes FP32 (ước tính) |
|---|---|---:|---:|
| A | 128×128×64 | 4,194,304 | 6,488,064 |
| B | 128×128×64 | 4,194,304 | 8,585,216 |
| C | 128×128×32 | 2,097,152 | 3,342,336 |

Output Conv lớn nhất của C là 128×128×16 = 1,048,576 bytes FP32. Feature map
lớn nhất tổng thể nằm ở `upsample_8`, không phải output Conv.

Ước tính peak dùng batch 1, từng lớp xử lý cả frame, cấp output trước khi giải
phóng input hết lượt dùng; giữ noisy input đến global Add và các skip của B đến
Add tương ứng. JSON ghi timeline và ước tính theo FP32/INT16/INT8. Không giả định
Add in-place hay fused upsample; không gồm weights, line buffers, accumulators,
DMA, alignment, training gradients hoặc TensorFlow allocator workspace. Đây là
một lịch cấp phát tham khảo, không phải tổng BRAM thực tế, minimum memory hay
đo RAM của TensorFlow. RTL streaming/tiled có thể giảm buffers nhưng phải giữ
halo, padding và sự đồng bộ đường residual.

## Train bằng VS Code trên Windows

Dùng Python 3.12 64-bit và extension Python. Nếu đã có `.venv` cho A/B, dùng lại
interpreter; nếu chưa có, tạo môi trường và cài dependencies tại root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -rs
```

Trong VS Code chọn `.venv\Scripts\python.exe`. TensorFlow 2.20 native Windows
chạy CPU; GPU cần môi trường WSL2 phù hợp. Đặt đúng đường dẫn SIDD và **dùng lại
split đã dùng cho A/B**:

```powershell
$siddRoot = 'C:\datasets\SIDD_Medium_Srgb\Data'
if (-not (Test-Path runs\sidd_split.json)) { throw 'Cần split gốc của A/B; không tạo lại để so sánh.' }
.\.venv\Scripts\python.exe -m src.model --config configs\hardware_cae.yaml
.\.venv\Scripts\python.exe -m src.architecture_report --output runs\architecture_comparison
.\.venv\Scripts\python.exe -m src.train --config configs\hardware_cae.yaml --data "$siddRoot" --output runs\hardware_smoke_01 --smoke
```

Smoke là 1 epoch với tối đa 2 batches training và 2 ảnh validation, 1 patch/cặp.
Nó không dùng toàn bộ protocol training. Full config giữ 320 cặp (256/64),
128×128, 4 patches/cặp/epoch, batch 8, seed 42, Adam 1e-4, tối đa 100 epoch.
Pipeline kiểm tra split/seed/pair membership và số lượng; không ghi đè split gốc.
Chỉ tự chạy lệnh full dưới đây khi chủ động muốn train:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\hardware_cae.yaml --data "$siddRoot" --output runs\hardware_cae_01
# Sau gián đoạn, dùng cùng config/data/output/split:
.\.venv\Scripts\python.exe -m src.train --config configs\hardware_cae.yaml --data "$siddRoot" --output runs\hardware_cae_01 --resume
```

Output riêng tránh đè kết quả A/B. `best.keras` chọn theo validation PSNR;
`recovery/current` chọn checkpoint phục hồi epoch hoàn tất gần nhất, gồm model,
Adam slots/iterations/LR, counters early stopping/giảm LR, best-model snapshot
và CSV. Epoch chưa commit sẽ chạy lại; không phải mid-batch recovery. Cần ít
nhất một epoch đã commit để resume; nếu gián đoạn trước đó, chạy lại ở output
mới. Giữ cùng cấu hình (chỉ tổng epochs được đổi) và split khi resume.
`training_history.csv`, `history.json`, `quality_history.png` chứa PSNR/SSIM.
Chi tiết cơ chế chung: [MODEL_B.md](MODEL_B.md).

Đánh giá C và A/B bằng cùng split, tile/overlap và công thức metrics hiện có:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --model runs\hardware_cae_01\best.keras --split runs\sidd_split.json --data "$siddRoot" --output runs\hardware_cae_01\evaluation
```

Chưa có kết quả PSNR/SSIM trên SIDD thật để điền vào bảng so sánh.

## Export weights cho RTL

Sau training, xuất checkpoint Model C vào **thư mục mới**:

```powershell
.\.venv\Scripts\python.exe -m src.export_weights --model runs\hardware_cae_01\best.keras --output runs\hardware_cae_01\rtl_export
```

Script chỉ đọc checkpoint; kiểm tra đầy đủ graph C (bao gồm global residual,
không skip), không sửa weights, không cast sang số nguyên và không tự train.
Mỗi Conv có kernel/bias `.npy` và `.csv`; NumPy FP32 là bản tensor chính xác.
CSV gồm một dòng header `value_fp32`, sau đó một giá trị mỗi dòng với 9 chữ số
có nghĩa, flatten theo C-order. Các file được đánh số Conv 01…10 theo execution
order, Conv 5 có tên `bottleneck`.

- Tensor ảnh/feature: NHWC; kernel: **HWIO = [kh, kw, ci, co]**; bias: [co].
- Địa chỉ kernel flatten: `(((kh * Kw + kw) * Cin + ci) * Cout + co)`; `co` chạy
  nhanh nhất, tiếp đến `ci`, `kw`, `kh`. Không nhầm với OIHW.
- Convolution là cross-correlation; RTL không lật kernel kh/kw.
- `manifest.json` ghi shape, dtype, order, tên file, SHA-256 checkpoint/tensor
  files. `layers.json` ghi graph shapes/padding/activation phục vụ đối chiếu.
- `.npy` có header nên không nạp trực tiếp bằng `$readmemh`. Khi chuẩn bị `.mem`,
  phải chọn address mapping và numeric format của RTL. Nếu dùng FP32 bit-pattern,
  reinterpret bits float32 thành uint32 và ghi mỗi word 8 hex digits; không dùng
  phép ép số thực sang integer. Endianness, word packing và bus width phải chốt
  với testbench. Không có converter fixed-point chính thức ở bước này.

INT8/INT16 cần định nghĩa scale/zero point, rounding, saturation, bias/accumulator
width, calibration trên train data phù hợp, rồi đánh giá lại PSNR/SSIM trên cùng
protocol. Ước tính dung lượng phía trên không thực hiện các bước này.

Bước tiếp theo cho Verilog: chọn một lớp để đối chiếu float reference và RTL;
kiểm tra padding/stride/upsample/Add, sau đó chốt fixed-point và overflow policy,
cuối cùng synthesis/timing/resource reports cho KV260. Chưa viết RTL toàn model.
