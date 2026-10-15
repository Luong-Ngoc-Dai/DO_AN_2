# Model D — 7-Conv paper-guided CAE

Model D được tích hợp từ `model_d_addon.zip` do người dùng cung cấp, trên base
`feat/hardware-friendly-cae`. `src/model_d.py` và `configs/model_d.yaml` giữ
nguyên từng byte của ZIP; không chạy installer và không thiết kế lại kiến trúc.
Ghi chú paper trong source là nguồn cảm hứng, không phải tuyên bố tái lập paper.

`src/model.py` chỉ thêm dispatch `paper_guided_cae` sang
`build_paper_guided_cae()`. Hàm dựng và graph A/B/C giữ nguyên; checkpoint cũ
không cần migration. Model D dùng pipeline training, metrics, evaluation và
recovery hiện có.

## Kiến trúc và độ phức tạp

7 Conv2D 3×3 SAME, filters `[16,32,48,32,16,16,3]`, strides
`[1,2,2,1,1,1,1]`; hai nearest upsampling ×2 sau Conv3 và Conv5. ReLU ở sáu
Conv đầu, Conv7 linear. Output = noisy RGB + correction; không có encoder skip,
BatchNorm, attention hoặc ConvTranspose. Input trực tiếp có H/W chia hết cho 4.

`python -m src.report_model_d` tính từ graph/weights shapes, không dùng constants
thay cho tính toán. Với RGB 128×128:

| Conv2D | Parameters (kernel + bias) | Conv MACs | Conv FLOPs |
|---:|---:|---:|---:|
| 7 | 40,195 | 160,432,128 | 320,864,256 |

1 MAC = 2 FLOPs; không gồm bias/ReLU/upsample/global Add. Xem bản xuất
[complexity.json](model_d_architecture/complexity.json),
[layers.json](model_d_architecture/layers.json) và
[model_summary.txt](model_d_architecture/model_summary.txt).

## Sửa lỗi/tích hợp so với addon

- Sửa ghi `model_summary.txt` trong training và summary A/B/C sang UTF-8 tường
  minh. Report D vốn đã dùng UTF-8, được giữ lại. Tests mô phỏng default CP1252
  để bắt lỗi ghi ký tự box-drawing; chưa chạy trên Windows thật.
- Exporter ban đầu kiểm tra số lớp/filter/params nhưng vẫn có thể chấp nhận
  graph đổi activation hoặc interpolation mà giữ nguyên số params. Nay so sánh
  toàn graph với builder D gốc; bỏ điều kiện hardcode `40195`, kiểm tra weights
  FP32 hữu hạn. Test graph sai cùng params bị từ chối. Định dạng/tên `.npy` và
  manifest của addon được giữ nguyên, không quantize.
- Thêm kiểm tra D bắt buộc dùng split có sẵn; với 320 cặp phải là 256/64,
  seed/pair membership hợp lệ. Split dùng chung chỉ được đọc.
- Report D có hàm gọi được từ tests, in thêm FLOPs, báo lỗi CLI theo pipeline
  chung. Không đổi phép tính PSNR/SSIM/UQI hoặc quy trình evaluation hiện có.
- Mở rộng tests addon để kiểm tra checkpoint, optimizer recovery, UTF-8,
  export round-trip, evaluation sau smoke và backward compatibility qua suite cũ.

## Windows PowerShell / VS Code

Clone nhánh đã tích hợp vào thư mục mới để tránh đè các thí nghiệm A/B/C.
Dùng Python 3.12 64-bit (phiên bản đã chạy test); không cần addon installer nữa.

```powershell
git clone --branch feat/model-d-cae --single-branch https://github.com/Luong-Ngoc-Dai/DO_AN_2.git DO_AN_2-model-d
cd .\DO_AN_2-model-d
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -rs
.\.venv\Scripts\python.exe -m src.model --config configs\model_d.yaml
.\.venv\Scripts\python.exe -m src.report_model_d
```

Trong VS Code chọn interpreter `.venv\Scripts\python.exe`. Native Windows
TensorFlow 2.20 chạy CPU; môi trường này chỉ kiểm chứng CPU Linux, chưa kiểm
chứng GPU/Windows native.

Chỉnh hai đường dẫn dưới đây, copy **split gốc đã dùng cho A/B/C**. Không tạo
lại split để so sánh. Nếu đã có `runs/sidd_split.json`, giữ nguyên và kiểm tra
đó là split gốc thay vì copy đè.

```powershell
$siddRoot = 'C:\datasets\SIDD_Medium_Srgb\Data'
$originalSplit = 'C:\path\to\ModelA\runs\sidd_split.json'
New-Item -ItemType Directory -Force -Path .\runs | Out-Null
if (-not (Test-Path .\runs\sidd_split.json)) {
    Copy-Item $originalSplit .\runs\sidd_split.json
}
if (-not (Test-Path $siddRoot)) { throw 'Sai duong dan SIDD Data' }
if (-not (Test-Path .\runs\sidd_split.json)) { throw 'Thieu split goc A/B/C' }
```

Chạy smoke trước, dùng output mới nếu đã chạy một lần:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\model_d.yaml --data "$siddRoot" --split runs\sidd_split.json --output runs\model_d_smoke --smoke
```

Smoke tối đa hai batches training và hai ảnh validation, một epoch,
một patch/cặp. Kết quả ở `runs/model_d_smoke/smoke/`; đây không phải full training.

Train **tối đa 100 epochs** khi chủ động chạy lệnh này (early stopping có thể
dừng trước). YAML giữ patch 128×128, bốn patch/cặp/epoch, batch 8, seed 42,
Adam 1e-4, 256 train / 64 validation:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\model_d.yaml --data "$siddRoot" --split runs\sidd_split.json --output runs\model_d_100epochs
```

Resume dùng cùng config/data/output/split:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\model_d.yaml --data "$siddRoot" --split runs\sidd_split.json --output runs\model_d_100epochs --resume
```

`best.keras` chọn theo validation-crop PSNR. Checkpoint recovery riêng tại
`recovery/current` giữ model, Adam slots/iterations/LR, callback counters, best
snapshot và history ở epoch commit gần nhất. Epoch dang dở được chạy lại;
không phải mid-batch resume. Nếu chưa có epoch hoàn tất thì cần chạy lại vào
output mới. Không dùng hai tiến trình ghi chung output; `best.keras` một mình
không đủ phục hồi trạng thái training. History CSV/JSON và biểu đồ PSNR/SSIM
được lưu như A/B/C.

Đánh giá ảnh validation sau training, không đổi metrics/preprocessing:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --model runs\model_d_100epochs\best.keras --split runs\sidd_split.json --data "$siddRoot" --output runs\model_d_100epochs\evaluation
```

Đây là PNG validation theo split, không phải benchmark SIDD Validation 1,280
block. Checkout không có evaluator MAT/1,280-block; hướng dẫn addon nhắc tới
benchmark đó không làm nó có sẵn. Chưa có kết quả PSNR/SSIM SIDD thật cho D.

Export weights/biases FP32 vào thư mục **chưa tồn tại**:

```powershell
.\.venv\Scripts\python.exe -m src.export_model_d --model runs\model_d_100epochs\best.keras --output runs\model_d_100epochs\weights_export
```

Output gồm bảy `conv_XX_kernel.npy`, bảy `conv_XX_bias.npy` và `manifest.json`
ghi SHA-256 checkpoint/tensors. Kernel layout `[Kh,Kw,Cin,Cout]` (HWIO), bias
`[Cout]`, flatten C-order; TensorFlow cross-correlation không lật kernel. Không
cast số nguyên, không quantization, không tạo Verilog/FPGA bitstream.

Dataset, `.venv`, `runs`, checkpoint và exported/trained weights không được
commit. Giữ outputs trong `runs/` đã được `.gitignore` bảo vệ.
