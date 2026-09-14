# Hướng dẫn setup dữ liệu Việt Nam

Branch: `vn-data`. Mục tiêu: chuẩn bị dataset VN 5 lớp (motorcycle / car / bus / truck / bicycle) để train baseline và improved.

## Tổng quan pipeline

```
Bước 1: Download nguồn ─┐
                       ├─► data/raw/<source>/{images,labels}/
Bước 2: Prepare data ──┘
             │
             ▼
   data/processed/{images,labels}/{train,val,test}/
             │
             ▼
   Train baseline + improved
```

## Bước 1 — Download từng nguồn dữ liệu

### 1a. Roboflow "Vietnamese vehicle" (LÕI CHÍNH — bắt buộc)

**~1 547 ảnh, format YOLO sẵn, 5 lớp**. Miễn phí CC BY 4.0.

1. Vào https://universe.roboflow.com/car-classification/vietnamese-vehicle
2. Đăng nhập / đăng ký tài khoản Roboflow (miễn phí)
3. Click **Download Dataset** → chọn:
   - Format: **YOLOv8**
   - Split: **Show Download Code** (tùy chọn)
4. Chọn cách tải:
   - **Cách A — Zip file**: click "Download zip to computer" → tải zip về máy → giải nén
   - **Cách B — API (nhanh hơn)**: dùng snippet `roboflow` Python (cần API key)
5. Đặt vào đúng cấu trúc:
   ```
   data/raw/roboflow_vn/
   ├── images/         # tất cả file .jpg
   │   ├── img_001.jpg
   │   ├── img_002.jpg
   │   └── ...
   └── labels/         # tất cả file .txt YOLO format
       ├── img_001.txt
       ├── img_002.txt
       └── ...
   ```

**Lưu ý**: Roboflow thường tải về có sẵn split train/valid/test. Gộp lại thành 1 folder `images/` và 1 folder `labels/` — script `prepare_data.py` sẽ tự chia lại.

Command gộp sau khi giải nén (thay `<extracted_dir>` bằng thư mục sau khi unzip):

```bash
mkdir -p data/raw/roboflow_vn/{images,labels}
cd <extracted_dir>
find . -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
  -exec cp {} ~/yolov8_finetune/data/raw/roboflow_vn/images/ \;
find . -type f -iname "*.txt" ! -name "*.yaml" ! -name "README*" \
  -exec cp {} ~/yolov8_finetune/data/raw/roboflow_vn/labels/ \;
cd ~/yolov8_finetune
ls data/raw/roboflow_vn/images | wc -l    # phải ra ~1547
ls data/raw/roboflow_vn/labels | wc -l    # phải ra ~1547
```

### 1b. VisDrone (BỔ SUNG — khuyến nghị nếu muốn thêm mật độ cao)

**~10 000 ảnh, format tự nhiên khác YOLO — cần convert**.

1. Vào https://github.com/VisDrone/VisDrone-Dataset
2. Tải `VisDrone2019-DET-train.zip` (khoảng 1.44GB)
3. Tải thêm `VisDrone2019-DET-val.zip` (khoảng 78MB) nếu muốn thêm data
4. Giải nén — cấu trúc gốc:
   ```
   VisDrone2019-DET-train/
   ├── images/          # ảnh JPG
   └── annotations/     # nhãn dạng CSV VisDrone (không phải YOLO)
   ```
5. **Convert sang YOLO format** bằng script Ultralytics-style (mình sẽ viết ở giai đoạn sau nếu cần).

Format VisDrone gốc: `<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>`.
YOLO cần: `<class_id> <cx_normalized> <cy_normalized> <w_normalized> <h_normalized>`.

Đặt kết quả convert vào:
```
data/raw/visdrone/
├── images/
└── labels/
```

**Bỏ qua bước này nếu muốn nhanh** — chỉ cần Roboflow VN là đã có thể chạy 1 vòng đầy đủ.

### 1c. UIT-VinaDeveS22 (tùy chọn — CẦN xin tác giả)

**1 364 ảnh CCTV VN, 7 lớp, mất cân bằng nặng**. Không có link public.

1. Email tác giả: Trịnh Đăng Thịnh, Nguyễn Thanh Kiều (Trường Công nghệ Thông tin & Truyền thông, CTU).
2. Nói rõ mục đích học thuật (đồ án tốt nghiệp).
3. Nếu được cấp: giải nén vào `data/raw/uit_vinades22/`.

Có thể bỏ qua giai đoạn 1, dùng làm test set độc lập (out-of-distribution) sau.

### 1d. Dữ liệu tự thu thập (tùy chọn — bù các case đặc biệt)

Nếu muốn tăng cường:
- Ảnh xe đạp, xe cứu hoả, xe ba gác… (các lớp hiếm)
- Điều kiện đêm, mưa, kẹt xe
- Cắt từ YouTube dashcam, CCTV giao thông VN

Tự gán nhãn bằng **Roboflow Annotate** hoặc **CVAT** hoặc **LabelImg**. Export YOLO format.

Đặt vào `data/raw/self_collected/`. Nhớ thêm mapping trong `configs/class_mapping.json`:
```json
"self_collected": {
  "0": "motorcycle",
  "1": "car",
  ...
}
```

## Bước 2 — Prepare data (gộp, remap, split)

Sau khi có ít nhất 1 source trong `data/raw/`:

```bash
# Kiểm tra source đã có
ls data/raw/
# Ví dụ: roboflow_vn  visdrone

# Chạy prepare_data
python scripts/prepare_data.py

# Hoặc chỉ định source cụ thể + split khác mặc định
python scripts/prepare_data.py --sources roboflow_vn visdrone --split 0.7 0.2 0.1
```

Kết quả:
```
data/processed/
├── images/
│   ├── train/    # file rename: <source>__<original>.jpg
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Script cũng in **class distribution** — quan trọng để xác định mức độ mất cân bằng thực tế:

```
=== processed/train: 12345 objects ===
  [0] motorcycle    6234  (50.51%) #########################
  [1] car           4321  (35.00%) #################
  [2] bus            890  ( 7.21%) ###
  [3] truck          678  ( 5.49%) ##
  [4] bicycle        122  ( 0.99%)
```

Nếu class hiếm (bicycle < 1%) → **class-weighted BCE sẽ có tác dụng rõ rệt** (khác với DETRAC).

## Bước 3 — Train baseline + improved trên VN data

Dùng lại `train.sh` nhưng đổi DATA:

```bash
# Sửa config trong scripts/train.sh
sed -i 's|DATA=data/detrac.yaml|DATA=data/data.yaml|' scripts/train.sh
sed -i 's|ARCH=models/yolov8_improved/yolov8n-cbam-detrac.yaml|ARCH=models/yolov8_improved/yolov8n-cbam.yaml|' scripts/train.sh
sed -i 's|NAME=exp_full|NAME=vn_exp|' scripts/train.sh

# Xem lại config
sed -n '19,35p' scripts/train.sh

# Train baseline VN
tmux new -s train_vn
./scripts/train.sh baseline
# Ctrl+B → D

# Sau khi baseline xong (~2-3h vì dataset nhỏ hơn DETRAC), train improved
tmux attach -t train_vn
./scripts/train.sh improved
# Ctrl+B → D
```

## Bước 4 — Đánh giá + so sánh

```bash
python scripts/evaluate.py \
  --data-yaml data/data.yaml \
  --yolo-baseline runs/yolo_baseline/vn_exp/weights/best.pt \
  --yolo-improved runs/yolo_improved/vn_exp/weights/best.pt \
  --data-root data/processed
```

Ra `results/tables/comparison.csv` với per-class metrics cho VN data.

## Kỳ vọng khoa học

Với VN data (mất cân bằng lớp cao, xe máy chiếm ~50-60%, xe đạp < 1%):

- **Baseline** dự kiến: mAP@0.5 ~ 0.50-0.65 (tùy chất lượng data). Recall class hiếm rất thấp.
- **Improved (CBAM + CW)** dự kiến: mAP@0.5 tương đương hoặc +1-3 điểm, nhưng **recall xe đạp/xe cứu hoả sẽ tăng nhiều** (5-15 điểm) — đây là điểm cần highlight trong báo cáo.
- **Ablation `--no-class-weights`**: sẽ cho biết CBAM một mình có đóng góp không, hay chính CW là driver.

Nếu kết quả đúng như kỳ vọng → **hoàn tất luận điểm giai đoạn 2 của đề tài**: cải tiến hiệu quả **có điều kiện** — cụ thể là trên dataset mất cân bằng cao như giao thông VN thực tế.

## Trạng thái giai đoạn 1 (DETRAC) — đã hoàn tất

Xem [BAO_CAO_KET_QUA.md](BAO_CAO_KET_QUA.md).

Kết luận: cải tiến chưa thắng baseline trên DETRAC vì DETRAC vốn cân bằng và mật độ thấp. Kết quả này **không phủ nhận** giá trị lý thuyết của cải tiến, đúng hơn xác nhận rằng chọn đúng bối cảnh áp dụng là quyết định.

Giai đoạn 2 (VN data) sẽ chứng minh giá trị của cải tiến trong bối cảnh phù hợp — đó là toàn bộ mục đích của branch `vn-data`.
