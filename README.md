# YOLOv8 cải tiến — Phát hiện & phân loại phương tiện giao thông Việt Nam

Đề tài NCKH sinh viên: cải tiến YOLOv8 để phát hiện 5 loại phương tiện
(**motorcycle, car, bus, truck, bicycle**) trong điều kiện giao thông VN mật độ
cao, mất cân bằng lớp nặng.

Hai mô hình được so sánh:

1. **YOLOv8 baseline** — fine-tune `yolov8n.pt` với cấu hình mặc định.
2. **YOLOv8 cải tiến** — chèn module **CBAM** (attention nhẹ) vào 3 nhánh output
   của neck, + **class-weighted BCE** (hoặc **Focal Loss**) để xử lý mất cân
   bằng lớp thiểu số (bicycle, truck).

---

## 1. Cấu trúc thư mục

```
yolov8_finetune/
├── data/
│   ├── raw/                    # dataset gốc (mỗi nguồn 1 subfolder)
│   ├── processed/              # sau khi chạy prepare_data.py
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   └── data.yaml               # khai báo 5 class cho Ultralytics
├── configs/
│   └── class_mapping.json      # ánh xạ class-id gốc → lớp chuẩn
├── models/
│   ├── yolov8_baseline/        # (placeholder, weight xuất về runs/)
│   └── yolov8_improved/
│       └── yolov8-cbam.yaml    # yaml kiến trúc + CBAM ở neck
├── scripts/
│   ├── prepare_data.py         # gộp, remap class, split, in stats
│   ├── train_yolo_baseline.py
│   ├── train_yolo_improved.py  # CBAM + class-weighted / focal
│   ├── evaluate.py             # so sánh 2 model → CSV + figures
│   └── demo.py                 # ảnh/video + đếm phương tiện
├── notebooks/
│   └── yolov8_finetune_pipeline.ipynb   # pipeline end-to-end có visualize từng bước
├── results/
│   ├── logs/                   # log train
│   ├── tables/                 # CSV: class distribution, comparison
│   └── figures/                # biểu đồ so sánh
├── weights/                    # yolov8n.pt / .pt tải sẵn
├── runs/                       # output train của Ultralytics
└── requirements.txt
```

---

## 2. Cài đặt (WSL2 + NVIDIA GPU)

Đã có conda env `yolov8_ft` (Python 3.10). Kích hoạt:

```bash
conda activate yolov8_ft
```

Nếu tạo mới từ đầu:

```bash
conda create -n yolov8_ft python=3.10 -y
conda activate yolov8_ft

# Torch bản CUDA 12.4 (khớp driver 5xx trong WSL)
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1

# Còn lại
pip install -r requirements.txt
```

Kiểm tra CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA RTX 3500 Ada Generation Laptop GPU
```

`nvidia-smi` phải hiển thị GPU. Không cần cài CUDA toolkit hệ thống — PyTorch
wheels tự bundle CUDA runtime.

---

## 3. Quy trình end-to-end

> **Muốn hiểu pipeline có visualize từng bước?** Mở
> [notebooks/yolov8_finetune_pipeline.ipynb](notebooks/yolov8_finetune_pipeline.ipynb)
> — chạy toàn bộ trên UA-DETRAC (Kaggle `bratjay/ua-detrac-orig`): tải dataset →
> parse XML → convert YOLO → train baseline & improved → so sánh.

### 3.1. Chuẩn bị dữ liệu

Đặt các dataset gốc vào `data/raw/<source_name>/` theo layout:

```
data/raw/roboflow_vn/
├── images/  *.jpg
└── labels/  *.txt   (định dạng YOLO)
data/raw/visdrone/
├── images/  *.jpg
└── labels/  *.txt
```

- **Roboflow "Vietnamese vehicle"**: tải từ Roboflow ở định dạng YOLOv8, giải
  nén ảnh + nhãn vào `data/raw/roboflow_vn/`. Nếu thứ tự lớp khác 5 lớp chuẩn
  → sửa `configs/class_mapping.json`.
- **VisDrone (một phần)**: chuyển sang YOLO format trước (ví dụ dùng script
  `visdrone2yolo`), rồi bỏ vào `data/raw/visdrone/`. Mapping mặc định đã có
  trong `configs/class_mapping.json` (chỉ giữ các lớp phương tiện).

Chạy:

```bash
python scripts/prepare_data.py --mapping configs/class_mapping.json
# hoặc:
python scripts/prepare_data.py --mapping configs/class_mapping.json --copy   # copy thay vì symlink
python scripts/prepare_data.py --stats-only                                    # chỉ in stats
```

Kết quả:
- `data/processed/{images,labels}/{train,val,test}/`
- `results/tables/class_distribution.csv` (đánh giá mức độ mất cân bằng).

### 3.2. Train

```bash
# 1) YOLOv8 baseline (mặc định)
python scripts/train_yolo_baseline.py --epochs 100 --batch 16

# 2) YOLOv8 cải tiến (CBAM + class-weighted BCE)
python scripts/train_yolo_improved.py --epochs 100 --batch 16

# Biến thể: dùng Focal Loss
python scripts/train_yolo_improved.py --epochs 100 --focal --focal-gamma 1.5
```

Log & weight lưu:
- `runs/yolo_baseline/exp/weights/best.pt`
- `runs/yolo_improved/exp/weights/best.pt`

### 3.3. Evaluate & so sánh

```bash
python scripts/evaluate.py \
    --yolo-baseline runs/yolo_baseline/exp/weights/best.pt \
    --yolo-improved runs/yolo_improved/exp/weights/best.pt \
    --fps-video samples/traffic.mp4
```

Output:
- `results/tables/comparison.csv` — mAP50, mAP50-95, P/R/acc per-class, FPS.
- `results/figures/map50_bar.png`, `map5095_bar.png`, `per_class_accuracy.png`.

### 3.4. Demo

```bash
# ảnh đơn
python scripts/demo.py --weights runs/yolo_improved/exp/weights/best.pt \
                       --source path/to/image.jpg --needs-cbam

# video
python scripts/demo.py --weights runs/yolo_improved/exp/weights/best.pt \
                       --source path/to/traffic.mp4 --needs-cbam \
                       --out results/demo/traffic_out.mp4

# thư mục ảnh
python scripts/demo.py --weights runs/yolo_improved/exp/weights/best.pt \
                       --source data/processed/images/test --needs-cbam \
                       --out results/demo/test_out
```

Cờ `--needs-cbam` bắt buộc khi weight là YOLOv8 cải tiến (để đăng ký
`NeckCBAM` trước khi load checkpoint). Với baseline (không CBAM) không cần cờ.

---

## 4. Ghi chú kỹ thuật

- **CBAM**: dùng `ultralytics.nn.modules.conv.CBAM` (đã có sẵn), bọc trong
  wrapper `NeckCBAM(c1, c2, k=7)` để tương thích với `parse_model`. Thêm 3 CBAM
  sau 3 output của neck (P3/P4/P5). Thay đổi param: ~3.0M → ~3.1M (rất nhẹ).
- **Class-weighted loss**: Ultralytics 8.4.x có sẵn `model.class_weights` được
  nhân vào BCE cls loss. Trainer tự tính từ `data/processed/labels/train`
  bằng công thức `w_c = N / (K · n_c)` (chuẩn hoá mean = 1).
- **Focal Loss**: monkey-patch `v8DetectionLoss.bce` bằng Focal BCE
  (gamma, alpha) — giữ nguyên shape `(bs, na, nc)` để tương thích logic hiện có.
- **Mất cân bằng dữ liệu**: xe máy thường >70%, xe đạp/xe tải <5%. Kỳ vọng
  cải tiến giúp tăng recall của bicycle và truck rõ nhất.

## 5. Troubleshooting WSL

- `libcudnn.so.9: cannot open shared object file` → cài lại torch với deps
  đầy đủ (bỏ `--no-deps`).
- `undefined symbol: iJIT_NotifyEvent` → torch bản conda-forge bị MKL
  mismatch; cài torch từ index chính thức của PyTorch như trong `README §2`.
- OOM trên GPU 12GB → giảm `--batch 8` hoặc `--imgsz 512`.
