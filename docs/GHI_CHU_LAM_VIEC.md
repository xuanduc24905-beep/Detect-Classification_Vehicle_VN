# GHI CHÚ LÀM VIỆC — Đồ án YOLOv8 phát hiện phương tiện giao thông Việt Nam

> File này là tài liệu **làm việc nội bộ**: khảo sát nghiên cứu liên quan, khoảng trống, phương
> pháp đề xuất, tình trạng triển khai hiện tại, và hướng phát triển tiếp theo. Dùng để tra cứu
> khi viết báo cáo, trả lời cô hướng dẫn, và định hướng công việc các tuần tới.

**Đề tài**: Phát hiện, phân loại phương tiện giao thông Việt Nam bằng YOLOv8 cải tiến
**Repo**: `~/yolov8_finetune`
**Ngày cập nhật**: 2026-09-12

---

## Phần I — Bối cảnh khoa học

### 1. Các nghiên cứu liên quan đã khảo sát

Bảng dưới tổng hợp 7 nguồn đã đọc, để tiện tra cứu khi viết báo cáo hoặc trả lời câu hỏi của cô.

| STT | Nghiên cứu | Hướng / Phương pháp | Dữ liệu | Kết quả chính & ghi chú dùng cho đề tài |
|---|---|---|---|---|
| 1 | Jocher, Chaurasia, Qiu (2023) — **Ultralytics** | Framework nền YOLOv8 (backbone CSPDarknet, neck PAN-FPN, head anchor-free) | — | Dùng làm nền tảng kỹ thuật, không phải nguồn khoảng trống. |
| 2 | Wen et al. (2020) — **UA-DETRAC** | Benchmark phát hiện/theo vết đa vật thể | Video giao thông Trung Quốc, ô tô/xe tải chủ đạo | Đại diện nhóm dữ liệu "dễ" — mật độ thấp, ít mất cân bằng. Dùng để so sánh, chỉ ra khác biệt với VN. |
| 3 | Tao (2026) — **Informatica** | ContextECA2.0 + Adaptive Window Attention trên YOLOv8n | VisDrone (ảnh drone, giao thông đông đúc) | mAP@0.5 +2.2%, mAP@0.5:0.95 +1.5%, 3.7M params, 120 FPS. → Bằng chứng: gắn attention giúp cảnh mật độ cao, nhưng **chưa test trên xe máy VN**. |
| 4 | Khalili & Smyth (2024) — **SOD-YOLOv8** | GFPN (tầng phát hiện thêm) + attention EMA trong C2f + Powerful-IoU loss | VisDrone + camera giao thông Columbia, Mỹ | Precision 51.2%→53.9%, Recall 40.1%→43.9%, mAP@0.5 40.6%→45.1%. → Bài mẫu dễ đọc để hiểu cách chèn attention vào YOLOv8. |
| 5 | Trịnh & Nguyễn (2022) — **UIT-VinaDeveS22** | So sánh YOLOX/YOLOF/YOLACT/RetinaNet | 1.364 ảnh CCTV Việt Nam, 7 lớp, ngày/đêm/mưa | Mất cân bằng cực nặng: **9.458 xe máy vs 233 xe đạp**. mAP cao nhất chỉ **0.306**. → Bằng chứng mạnh nhất cho khoảng trống "VN khó hơn benchmark quốc tế". |
| 6 | Nguyễn & Phạm (2025) — **Springer ICTA** | YOLOv8 + StrongSORT/ByteTrack (tracking) | Dữ liệu xe VN tự thu thập (chưa công khai) | mAP phát hiện 0.87. Tập trung tracking, **KHÔNG** xử lý mất cân bằng lớp hay cải tiến kiến trúc. |
| 7 | Nguyễn và cộng sự (2024) — **ACM ICIIT** | So sánh mô hình nhận diện biển báo | VTSDB46 — biển báo giao thông VN, 46 lớp | Không phải về xe, nhưng cho thấy cộng đồng NC giao thông VN đang hoạt động — dùng làm bối cảnh, không phải khoảng trống. |

### 2. Khoảng trống nghiên cứu (4 điểm)

1. Fine-tune YOLOv8 "nguyên bản" (hướng ban đầu của đề tài) mới được kiểm chứng trên dữ liệu **mật độ thấp, cân bằng lớp tốt** hơn (UA-DETRAC…) — chưa chắc đúng khi áp vào VN.
2. Các cải tiến **attention** cho YOLOv8 (ContextECA2.0, SOD-YOLOv8) đã chứng minh hiệu quả trên giao thông mật độ cao nói chung, nhưng **chưa từng test trên dữ liệu xe máy Việt Nam**.
3. Bộ dữ liệu VN duy nhất (UIT-VinaDeveS22) cho thấy **mất cân bằng lớp cực nặng** và mAP rất thấp (0.17–0.31), nhưng chưa ai xử lý mất cân bằng này hay báo cáo accuracy riêng từng lớp.
4. Chưa có nghiên cứu nào **kết hợp đồng thời**: cải tiến kiến trúc (attention) + xử lý mất cân bằng lớp + đánh giá theo điều kiện môi trường (ngày/đêm/mưa) trên **cùng bối cảnh giao thông VN**.

### 3. Hướng mới để triển khai (phương pháp đề xuất)

- **Dữ liệu**: kết hợp dữ liệu công khai (xem mục 4) với ảnh/video VN tự thu thập; thống kê phân bố lớp ngay từ đầu để biết mức độ mất cân bằng cụ thể.
- **Mô hình**: giữ 2 mô hình đối chứng theo đề tài gốc (CNN from scratch, YOLOv8 fine-tune thường) và thêm mô hình thứ 3 — **YOLOv8 cải tiến** — chèn module attention nhẹ (ECA hoặc **CBAM**, có sẵn qua cấu hình Ultralytics, không cần tự thiết kế) vào neck, kết hợp **loss có trọng số lớp (class-weighted / focal loss)** để cải thiện các lớp thiểu số.
- **Đánh giá**: so sánh 3 mô hình trên cùng dữ liệu bằng `mAP@0.5`, `mAP@0.5:0.95`, precision/recall/accuracy riêng từng lớp (không chỉ số tổng), và FPS. Nếu thu thập được dữ liệu ngày/đêm/mưa thì báo cáo thêm theo từng điều kiện.
- **Mức độ đóng góp**: áp dụng có chọn lọc kỹ thuật đã kiểm chứng (attention cho vật thể nhỏ/che khuất) vào đúng bối cảnh còn bỏ ngỏ (giao thông VN mất cân bằng lớp cao) — vừa đủ mới, vừa khả thi trong 2–3 tháng.

### 4. Nguồn dữ liệu

| Nguồn | Tình trạng | Nội dung | Ghi chú |
|---|---|---|---|
| **Roboflow "Vietnamese vehicle"** | Tải trực tiếp, miễn phí (CC BY 4.0) | 1.547 ảnh — car/bus/truck/motorcycle, format YOLO sẵn | Lõi chính cho tập huấn luyện — [universe.roboflow.com/car-classification/vietnamese-vehicle](https://universe.roboflow.com/car-classification/vietnamese-vehicle) |
| **VisDrone-Dataset** | Tải trực tiếp, public | Ảnh/video drone, giao thông mật độ cao | Bổ sung cảnh đông đúc để huấn luyện phần attention; [github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) |
| **UA-DETRAC** | Tải được (form hoặc Kaggle/Roboflow mirror) | Video giao thông, ô tô/xe tải chủ đạo | Dùng làm baseline đối chứng "data dễ" nếu cần |
| **UIT-VinaDeveS22** | **KHÔNG** có link công khai | 1.364 ảnh CCTV VN, 7 lớp, mất cân bằng nặng | Cần email xin tác giả (Trịnh, Nguyễn — CTU) cho mục đích học thuật |
| **Tự thu thập bổ sung** | Tự làm | Ảnh/video giao thông VN (dashcam, CCTV, cắt YouTube) | Bù các lớp thiếu (xe đạp, xe cứu hoả) và điều kiện đêm/mưa |

---

## Phần II — Trạng thái triển khai hiện tại

### 5. Kiến trúc phần mềm (pipeline)

Repo tổ chức theo mô hình pipeline: mỗi script làm 1 stage, giao tiếp qua file trên disk.

```
configs/class_mapping.json
        │
        ▼
[prepare_data.py] ──► data/processed/{images,labels}/{train,val,test} + data.yaml
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                      ▼
   [train_yolo_baseline.py]                    [train_yolo_improved.py]
   runs/yolo_baseline/exp/                     runs/yolo_improved/exp/
        weights/{best,last}.pt                     weights/{best,last}.pt
                    │                                      │
                    └──────────────┬───────────────────────┘
                                   ▼
                          [evaluate.py]  ──►  results/tables/*.csv, results/figures/*.png
                                   │
                                   ▼
                          [demo.py]  ──►  results/demo/*.mp4
```

Chi tiết vai trò từng file: xem [scripts/](../scripts/).

### 6. Hai cải tiến đang thử nghiệm

Thực hiện trong [scripts/train_yolo_improved.py](../scripts/train_yolo_improved.py):

#### 6.1. **CBAM ở neck** (Convolutional Block Attention Module)

- File kiến trúc: [models/yolov8_improved/yolov8n-cbam.yaml](../models/yolov8_improved/yolov8n-cbam.yaml) và biến thể DETRAC [yolov8n-cbam-detrac.yaml](../models/yolov8_improved/yolov8n-cbam-detrac.yaml) (nc=4).
- **Nơi chèn**: sau các block C2f cuối cùng của neck, ngay trước 3 head detect (P3/P4/P5). Cả 3 scale detect đều được "re-weight" theo channel + spatial attention.
- **Cơ chế**: CBAM = Channel Attention (nén H×W thành 1×1, học trọng số kênh) + Spatial Attention (nén C thành 1, học heatmap không gian) — nhân lần lượt vào feature map.
- **Chi phí**: thêm ~0.1M params (yolov8n gốc 3.0M → improved 3.09M), ~8.2 GFLOPs (tăng không đáng kể).
- **Kỳ vọng**: mô hình chú ý vào vùng có xe (small objects + cluttered background) tốt hơn.
- **Cách patch vào Ultralytics**: hàm `register_cbam()` trong script monkey-patch `ultralytics.nn.tasks.parse_model` để thêm `NeckCBAM` vào frozenset `base_modules` — nhờ đó `parse_model` đọc được yaml có tag `NeckCBAM`.

#### 6.2. **Class-weighted BCE loss**

- Hàm `compute_class_weights()` đếm số nhãn mỗi lớp trong `labels/train/`, tính:
  ```
  w[i] = N_total / (K * n_i)     # chuẩn hoá về mean = 1
  ```
  (N_total = tổng số nhãn, K = số lớp, n_i = số nhãn lớp i).
- Gán `model.model.class_weights = w` → Ultralytics 8.4.x **tự nhân trọng số này vào BCE cls loss**.
- **Kỳ vọng**: recall các lớp hiếm (xe đạp, xe cứu hoả) tăng, ít bị model bỏ qua.

#### 6.3. **(Tuỳ chọn) Focal Loss** — cờ `--focal`

- Hàm `patch_focal_loss(gamma, alpha)` monkey-patch `v8DetectionLoss.__init__`, thay `BCEWithLogitsLoss` bằng `_FocalBCE` giữ nguyên chuẩn hoá + class_weights.
- Mặc định `gamma=1.5`, `alpha=0.25` — nhấn mạnh hard examples.
- **Dùng khi**: class-weighted BCE vẫn chưa đủ, có vài lớp bị predict với confidence rất thấp.

### 7. Trạng thái thực nghiệm (12/09/2026)

Cấu hình chung: **DETRAC** (nc=4), imgsz=640, batch=16, `patience=15`, `--cache ram` (fallback do 51GB RAM không đủ 87.5GB).

| Run | Đường dẫn | Epoch đã train | Best epoch | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| Baseline yolov8n | `runs/yolo_baseline/exp/` | 18 / 50 | ep. 3 | **0.816** | **0.622** |
| Improved yolov8n+CBAM+CW | `runs/detect/runs/yolo_improved/exp/` | 20 / 50 | ep. 5 | 0.801 | 0.615 |
| Improved (run cũ, đã dừng) | `runs/yolo_improved/exp/` | 13 / 50 | ep. 3 | 0.803 | 0.609 |

**Quan sát**:
- Cả 2 run **peak sớm (epoch 3–5)**, sau đó dao động — chưa hội tụ, cần train tiếp cho đủ 50 epoch (thời gian còn lại ~9h nếu resume tuần tự).
- Baseline hiện đang **hơn improved ~1.5 mAP** — chưa kết luận được vì improved bị dừng sớm hơn ở giai đoạn học kiến trúc CBAM.
- `--cache ram` không hoạt động trên máy 64GB (cần ~96GB) → epoch chậm ~600s (improved) và ~430s (baseline).

### 8. Vấn đề đã gặp / đã xử lý

| Vấn đề | Nguyên nhân | Đã xử lý |
|---|---|---|
| Script improved hard-code `num_classes=5` | Ban đầu viết cho 5 lớp VN vehicle | Đọc `nc` từ `data.yaml`; auto-suy labels_dir từ `path + train`. |
| Không có flag `--resume` cho improved | Chỉ baseline có | Thêm nhánh resume trong [train_yolo_improved.py](../scripts/train_yolo_improved.py), load thẳng `last.pt`. |
| RAM 51GB không cache nổi 87.5GB train | Ảnh 960×540 × 68k = quá to | Chuyển sang `--cache disk` (chỉ cần ổ trống 87GB). |
| Notebook "Run All" ghi đè run cũ | `exist_ok=True` trong cell 11 | Sửa ô train thành resume-aware (xem doc `NOTEBOOK_RESUME.md` nếu tách sau). |
| Path log bị nested `runs/detect/runs/...` | Ultralytics 8.4.x tạo thư mục con `detect/` khi task = detect | Không nghiêm trọng, chỉ cần biết đúng đường dẫn để tìm `best.pt`. |

### 9. Tooling đã có

- [scripts/train.sh](../scripts/train.sh): wrapper bash cho 3 mode `improved | resume | baseline`, hỗ trợ `bg` (nohup + log timestamp).
- [scripts/prepare_data.py](../scripts/prepare_data.py): gộp nhiều source, remap class-id, chia split.
- [scripts/evaluate.py](../scripts/evaluate.py): so sánh 2 model, xuất `results/tables/comparison.csv` + biểu đồ.
- [scripts/demo.py](../scripts/demo.py): inference video, đếm phương tiện.
- [scripts/gen_readme_docx.py](../scripts/gen_readme_docx.py): sinh `docs/README_YOLOv8_FineTune.docx` từ template.

---

## Phần III — Kế hoạch phát triển tiếp

### 10. Việc cần làm ngay (tuần này)

- [ ] **Resume train baseline** cho đủ 50 epoch (`./scripts/train.sh baseline bg` với `--resume` — cần thêm nhánh resume cho baseline nếu chưa).
- [ ] **Resume train improved** cho đủ 50 epoch (`./scripts/train.sh resume bg`).
- [ ] Đổi `CACHE=ram` → `CACHE=disk` trong [train.sh](../scripts/train.sh) để giảm thời gian epoch xuống ~450s.
- [ ] Sau khi cả 2 xong: `python scripts/evaluate.py` → xem `results/tables/comparison.csv` và **per-class metrics**.
- [ ] Kiểm tra `runs/*/exp/labels.jpg` (phân bố nhãn) và `confusion_matrix.png` để xác nhận mức độ mất cân bằng thực tế của DETRAC (dataset "dễ", có thể mAP không lộ ra vấn đề class-weight).

### 11. Vấn đề khoa học chưa giải quyết

**DETRAC là dataset "dễ"** so với kịch bản đề tài đề ra (giao thông VN mất cân bằng). Nếu chỉ train + evaluate trên DETRAC:
- Không chứng minh được đóng góp cho khoảng trống #3 và #4 (mất cân bằng lớp VN).
- Kết quả CBAM có thể **không cải thiện đáng kể** vì DETRAC vốn ít vật thể nhỏ/che khuất so với VisDrone.

**→ Bắt buộc phải chuyển sang dữ liệu VN** (Roboflow Vietnamese vehicle + tự thu thập) ở giai đoạn 2, không dừng ở DETRAC.

### 12. Ma trận cải tiến — sắp xếp theo ROI

Sau khi có số per-class từ evaluate, chọn 2–3 hướng dưới để làm tiếp:

| Hướng | Chi phí công | Kỳ vọng cải thiện | Khi nào chọn |
|---|---|---|---|
| **Chuyển sang dữ liệu VN (Roboflow + tự thu thập)** | 1–2 tuần data work | **Bắt buộc** — không thì không đóng góp gì cho khoảng trống VN | Ngay sau khi eval DETRAC xong |
| **Scale to yolov8s** | Đổi 1 flag, train lại | +2–4 mAP@0.5 gần như chắc chắn | Khi VRAM đủ, muốn tăng chất lượng nhanh |
| **Tăng imgsz 640→960** | 1 flag, epoch chậm 2x | +1–3 mAP@0.5 cho object nhỏ (biển báo, xe xa) | Khi giao thông có nhiều xe xa/nhỏ |
| **Focal Loss** | Cờ `--focal`, ablation | +0.5–1.5 mAP cho class hiếm | Nếu evaluate cho thấy recall lớp hiếm còn thấp |
| **YOLOv11n baseline** | Đổi weights + viết yolo11-cbam.yaml | +1–2 mAP, có cột tham chiếu version mới trong báo cáo | Khi muốn thể hiện "đã khảo sát version mới" |
| **Progressive resize** (train 320 → fine-tune 640) | 2 pass train, tổng thời gian tương đương | +0.5–1 mAP, hội tụ nhanh giai đoạn đầu | Muốn tận dụng cache RAM ở res thấp |
| **Data cleaning** (nhãn nhiễu, hard negatives) | 1–2 ngày | +2–5 mAP nếu label noise cao | Khi evaluate lộ ra class có precision bất thường thấp |
| **Đánh giá theo điều kiện môi trường** (ngày/đêm/mưa) | Cần metadata split; nếu đã có → 1 ngày | Không tăng mAP, nhưng **thoả yêu cầu khoảng trống #4** | Sau khi có dữ liệu VN có metadata |

### 13. Timeline gợi ý (2–3 tháng)

| Tuần | Việc chính | Output kỳ vọng |
|---|---|---|
| 1 | Resume train DETRAC baseline + improved, evaluate | Bảng per-class DETRAC, confusion matrix |
| 2 | Tải + chuẩn bị Roboflow VN + VisDrone, gộp với DETRAC | Dataset gộp có metadata source |
| 3 | Xin UIT-VinaDeveS22, tự thu thập bổ sung | Có ít nhất 3k ảnh VN với metadata môi trường |
| 4–5 | Train baseline + improved trên dataset VN | mAP + per-class trên tập VN |
| 6 | Ablation: `--focal`, `--scale s`, `imgsz 960` | 3–5 bảng ablation |
| 7 | Đánh giá điều kiện ngày/đêm/mưa | Bảng theo môi trường |
| 8 | Thêm YOLOv11 làm cột tham chiếu | Bảng so sánh 4 model |
| 9–10 | Viết báo cáo + polish demo | Bản nháp báo cáo + video demo |
| 11–12 | Sửa theo feedback cô, hoàn thiện | Nộp bản cuối |

### 14. Rủi ro và mitigations

| Rủi ro | Mức | Mitigations |
|---|---|---|
| UIT-VinaDeveS22 không xin được | Trung | Không phụ thuộc — dùng Roboflow VN + tự thu thập là đủ. UIT chỉ là "nice to have" cho tham chiếu. |
| Roboflow VN quá ít (1.5k) → overfit | Cao | Bắt buộc combine với VisDrone và tự thu thập. Có augmentation mạnh (mosaic, mixup). |
| CBAM không cải thiện trên VN (do đã có augment) | Trung | Fallback: chỉ dùng class-weighted BCE + focal, vẫn có đóng góp cho khoảng trống #3. |
| Không đủ thời gian train nhiều run | Cao | Ưu tiên: 1 baseline + 1 improved + 1–2 ablation. Không tham hàng chục variant. |
| Máy 64GB không đủ RAM cho dataset lớn hơn | Trung | `--cache disk`, hoặc `--fraction 0.7`, hoặc mượn máy lab. |

### 15. Log các quyết định quan trọng

- **[2026-09-xx]** Chọn CBAM thay ECA vì CBAM đã có sẵn trong Ultralytics, không cần tự implement — giảm rủi ro bug.
- **[2026-09-xx]** Dùng DETRAC làm dataset "training thử" trước, không phải dataset chính — để pipeline chạy được, ăn số trước khi chuyển sang VN data.
- **[2026-09-12]** Fix hard-code `num_classes=5` trong script improved để dùng được cho cả VN vehicle (5) và DETRAC (4).
- **[2026-09-12]** Thêm flag `--resume` cho improved script. Trước đó chỉ baseline có.

---

## Phụ lục A — Lệnh chạy thường dùng

```bash
# Chuẩn bị data (chạy 1 lần cho VN dataset khi có)
python scripts/prepare_data.py

# Train baseline (foreground / background)
./scripts/train.sh baseline
./scripts/train.sh baseline bg

# Train improved (foreground / background)
./scripts/train.sh improved
./scripts/train.sh improved bg

# Resume improved từ last.pt
./scripts/train.sh resume bg

# Theo dõi log run mới nhất
tail -f logs/$(ls -t logs/ | head -1)

# Theo dõi GPU
watch -n 2 nvidia-smi

# Đo tốc độ epoch thực tế
awk -F, 'NR>1{print $1, $2-prev; prev=$2}' runs/detect/runs/yolo_improved/exp/results.csv | tail -5

# So sánh 2 model
python scripts/evaluate.py

# Demo trên video
python scripts/demo.py \
  --weights runs/yolo_improved/exp/weights/best.pt \
  --source samples/traffic.mp4 \
  --out results/demo/out.mp4

# Sinh lại README.docx
python scripts/gen_readme_docx.py
```

## Phụ lục B — Cấu trúc thư mục

```
yolov8_finetune/
├── configs/
│   └── class_mapping.json          # remap class-id giữa các source
├── data/
│   ├── detrac.yaml                 # config DETRAC (nc=4, đang test)
│   ├── data.yaml                   # config VN vehicle (nc=5, khi có)
│   ├── detrac/                     # DETRAC images + labels
│   └── processed/                  # data VN sau khi prepare
├── models/
│   └── yolov8_improved/
│       ├── yolov8n-cbam.yaml       # nc=5 (VN)
│       └── yolov8n-cbam-detrac.yaml # nc=4 (DETRAC)
├── weights/
│   └── yolov8n.pt                  # pretrained COCO
├── scripts/
│   ├── prepare_data.py
│   ├── train_yolo_baseline.py
│   ├── train_yolo_improved.py
│   ├── evaluate.py
│   ├── demo.py
│   ├── gen_readme_docx.py
│   └── train.sh                    # bash wrapper
├── notebooks/
│   └── yolov8_finetune_pipeline.ipynb  # phiên bản notebook đầy đủ
├── runs/
│   ├── yolo_baseline/exp/          # kết quả baseline
│   └── yolo_improved/exp/          # kết quả improved
├── logs/                           # log của train.sh bg
├── results/                        # bảng + figure so sánh
├── docs/
│   ├── README_YOLOv8_FineTune.docx
│   └── GHI_CHU_LAM_VIEC.md         # ← file này
└── requirements.txt
```
