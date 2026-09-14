---
title: |
    Báo cáo kết quả nghiên cứu
    Phát hiện và phân loại phương tiện giao thông
    bằng YOLOv8 cải tiến với CBAM và loss có trọng số lớp
subtitle: Thực nghiệm trên bộ dữ liệu UA-DETRAC
author: xuanduc24905
date: 2026-09-12
---

# Tóm tắt

Báo cáo trình bày kết quả giai đoạn 1 của đồ án phát hiện phương tiện giao thông
Việt Nam bằng YOLOv8 cải tiến. Chúng tôi đề xuất hai cải tiến so với baseline
YOLOv8n: (1) chèn module chú ý CBAM (Convolutional Block Attention Module) vào
neck của mạng và (2) sử dụng Binary Cross-Entropy có trọng số lớp (class-weighted
BCE) cho nhánh phân loại nhằm giảm ảnh hưởng của mất cân bằng lớp. Ở giai đoạn
thử nghiệm này, hai mô hình được huấn luyện trên tập UA-DETRAC (68 000 ảnh
huấn luyện, 4 lớp phương tiện) với cùng cấu hình siêu tham số. Kết quả cho thấy
cả hai mô hình đều hội tụ rất sớm (epoch 3–5) và dừng theo cơ chế early stopping.
Baseline đạt mAP@0.5 = 0.816 và mAP@0.5:0.95 = 0.622; mô hình cải tiến đạt
mAP@0.5 = 0.801 và mAP@0.5:0.95 = 0.615. Việc cải tiến chưa thắng baseline
trên UA-DETRAC — một bộ dữ liệu tương đối cân bằng — được thảo luận và đặt
làm tiền đề cho giai đoạn 2: thực nghiệm trên dữ liệu giao thông Việt Nam nơi
mất cân bằng lớp và mật độ vật thể cao hơn nhiều.

**Từ khoá**: YOLOv8, CBAM, class-weighted loss, phát hiện phương tiện, UA-DETRAC,
attention mechanism.

---

# 1. Giới thiệu

Phát hiện và phân loại phương tiện giao thông là bài toán nền tảng cho các hệ
thống giao thông thông minh (ITS): giám sát mật độ giao thông, đếm phương tiện,
phát hiện vi phạm và hỗ trợ điều khiển tín hiệu. Trong bối cảnh giao thông
Việt Nam, bài toán này có đặc thù riêng: mật độ phương tiện cao, tỉ lệ xe máy
lớn, mất cân bằng lớp nghiêm trọng (theo Trịnh & Nguyễn, 2022, tỉ lệ xe máy so
với xe đạp trong CCTV giao thông Việt Nam lên tới **40:1**), và điều kiện thu
thập đa dạng (ngày/đêm/mưa).

Đa số nghiên cứu tiên tiến hiện nay dựa trên họ mô hình YOLO. YOLOv8
(Jocher et al., 2023) là phiên bản phổ biến nhờ độ chính xác cao, tốc độ suy luận
tốt, hỗ trợ đầy đủ trong hệ sinh thái Ultralytics. Tuy nhiên, các nghiên cứu
cải tiến YOLOv8 dùng attention như ContextECA2.0 (Tao, 2026) hay SOD-YOLOv8
(Khalili & Smyth, 2024) mới chỉ được kiểm chứng trên VisDrone hoặc dữ liệu
giao thông quốc tế, chưa có báo cáo trên dữ liệu giao thông Việt Nam.

Đóng góp của báo cáo giai đoạn 1 này:

1. Xây dựng pipeline huấn luyện – đánh giá – so sánh có thể tái sử dụng cho
   cả dữ liệu quốc tế và dữ liệu Việt Nam.
2. Thực hiện hai cải tiến kỹ thuật (CBAM + class-weighted BCE) trên YOLOv8n
   và so sánh có kiểm soát với baseline trên UA-DETRAC.
3. Phân tích lý do cải tiến chưa thắng baseline trên bộ dữ liệu tương đối
   cân bằng, tạo cơ sở lý luận để chuyển sang giai đoạn 2 trên dữ liệu Việt
   Nam nơi cải tiến kỳ vọng phát huy tác dụng.

---

# 2. Nghiên cứu liên quan

Bảy nguồn được khảo sát và tóm tắt trong Bảng 1. Bảng này cũng chỉ ra các
"nhóm dữ liệu" và các "loại cải tiến" đã được cộng đồng nghiên cứu xử lý,
làm nền cho khoảng trống nghiên cứu ở mục 3.

**Bảng 1**. Tổng hợp nghiên cứu liên quan.

| # | Tác giả | Phương pháp | Dữ liệu | Kết quả chính |
|---|---|---|---|---|
| 1 | Jocher, Chaurasia, Qiu (2023) | Framework nền YOLOv8 (backbone CSPDarknet, neck PAN-FPN, head anchor-free) | — | Nền tảng kỹ thuật |
| 2 | Wen et al. (2020) — UA-DETRAC | Benchmark phát hiện/theo vết đa vật thể | Video giao thông TQ, chủ đạo ô tô/xe tải | Đại diện dữ liệu "dễ" — cân bằng lớp tốt hơn |
| 3 | Tao (2026) | ContextECA2.0 + Adaptive Window Attention trên YOLOv8n | VisDrone | mAP@0.5 +2.2%, mAP@0.5:0.95 +1.5%, 120 FPS |
| 4 | Khalili & Smyth (2024) — SOD-YOLOv8 | GFPN + EMA attention trong C2f + Powerful-IoU | VisDrone + CCTV Columbia | mAP@0.5 40.6% → 45.1% |
| 5 | Trịnh & Nguyễn (2022) — UIT-VinaDeveS22 | So sánh YOLOX/YOLOF/YOLACT/RetinaNet | 1 364 ảnh CCTV VN, 7 lớp | mAP cao nhất 0.306; mất cân bằng 40:1 |
| 6 | Nguyễn & Phạm (2025) | YOLOv8 + StrongSORT/ByteTrack | Dữ liệu VN tự thu thập | mAP phát hiện 0.87, tập trung tracking |
| 7 | Nguyễn et al. (2024) | So sánh mô hình nhận diện biển báo | VTSDB46, 46 lớp | Bối cảnh nghiên cứu giao thông VN |

---

# 3. Khoảng trống nghiên cứu

Từ Bảng 1, chúng tôi xác định bốn khoảng trống chính:

1. **Fine-tune YOLOv8 "nguyên bản"** mới được kiểm chứng trên dữ liệu mật độ
   thấp, cân bằng lớp tốt (UA-DETRAC…) — chưa chắc phù hợp trực tiếp với dữ
   liệu Việt Nam.
2. **Các cải tiến attention cho YOLOv8** (ContextECA2.0, SOD-YOLOv8) đã chứng
   minh hiệu quả trên giao thông mật độ cao nói chung nhưng chưa từng được
   thử nghiệm trên dữ liệu xe máy Việt Nam.
3. **Bộ dữ liệu VN duy nhất** (UIT-VinaDeveS22) cho thấy mất cân bằng lớp cực
   nặng và mAP thấp (0.17–0.31), nhưng chưa có nghiên cứu xử lý mất cân bằng
   này hay báo cáo accuracy riêng từng lớp.
4. **Chưa có nghiên cứu nào kết hợp đồng thời**: cải tiến kiến trúc (attention)
   + xử lý mất cân bằng lớp + đánh giá theo điều kiện môi trường (ngày/đêm/mưa)
   trên cùng bối cảnh giao thông Việt Nam.

Báo cáo giai đoạn 1 này xử lý trực tiếp các khoảng trống #1 và #2 (kiểm chứng
attention + fine-tune YOLOv8 trên benchmark quốc tế) như bước đệm cho giai đoạn 2
xử lý khoảng trống #3, #4 trên dữ liệu Việt Nam.

---

# 4. Phương pháp

## 4.1. Kiến trúc mô hình cải tiến — YOLOv8n + CBAM

Chúng tôi giữ nguyên backbone và head của YOLOv8n, chèn module CBAM
(Convolutional Block Attention Module — Woo et al., 2018) vào phần neck. Cụ thể,
sau các block C2f cuối cùng của neck ứng với ba scale phát hiện P3, P4, P5,
một khối CBAM được thêm vào ngay trước lớp `Detect`. Cả ba đầu ra phát hiện đều
được "tái trọng số" theo hai chiều:

- **Channel attention**: nén đặc trưng H×W thành 1×1 qua global pooling, học
  vector trọng số kích cỡ C bằng MLP hai lớp, nhân trở lại vào feature map theo
  chiều kênh.
- **Spatial attention**: nén đặc trưng theo chiều kênh về 1, học một heatmap
  H×W bằng convolution 7×7, nhân trở lại vào feature map theo chiều không gian.

Chi phí kiến trúc:

- Tham số: 3.00 M (baseline) → **3.10 M** (+ ~0.1 M, tức + 3.3 %).
- GFLOPs @ 640×640: 8.1 → **8.2** (không đáng kể).
- Số lớp: 129 → **144** (thêm 15 lớp CBAM cho 3 vị trí × 5 lớp mỗi khối).

Trong Ultralytics 8.4.x, module CBAM đã có sẵn (`ultralytics.nn.modules.conv.CBAM`).
Chúng tôi định nghĩa lớp bao ngoài `NeckCBAM(c1, c2, k=7)` để tương thích với
signature của hàm `parse_model` của Ultralytics và đăng ký lớp này vào
`base_modules` bằng cơ chế monkey-patch. Kiến trúc CBAM được khai báo trong tệp
`models/yolov8_improved/yolov8n-cbam-detrac.yaml`.

## 4.2. Class-weighted BCE cho nhánh phân loại

Ultralytics 8.4.x hỗ trợ trọng số lớp trong Binary Cross-Entropy của nhánh
`cls` thông qua thuộc tính `model.class_weights`. Chúng tôi khai báo trọng số
theo tần suất nghịch đảo được chuẩn hoá về trung bình bằng 1:

$$
w_i = \frac{N_{\text{total}}}{K \cdot n_i}
$$

trong đó $N_{\text{total}}$ là tổng số nhãn của tập huấn luyện,
$K$ là số lớp, $n_i$ là số nhãn lớp $i$. Chuẩn hoá về mean = 1
đảm bảo độ lớn tổng của loss không đổi so với BCE thường, giữ scale ổn định
cho LR schedule.

## 4.3. (Tuỳ chọn) Focal Loss

Ngoài class-weighted BCE, chúng tôi triển khai tuỳ chọn thay BCE bằng
Focal Loss (Lin et al., 2017) qua monkey-patch `v8DetectionLoss.__init__`.
Focal Loss có dạng:

$$
\mathcal{L}_{\text{focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

với $\gamma = 1.5$, $\alpha = 0.25$ mặc định. Chức năng này chưa được đánh giá
trong báo cáo giai đoạn 1 và sẽ được sử dụng trong giai đoạn 2 nếu class-weighted
BCE thuần chưa đủ.

---

# 5. Thực nghiệm

## 5.1. Dữ liệu

Bộ dữ liệu **UA-DETRAC** (Wen et al., 2020) — 4 lớp phương tiện, phân chia
theo Bảng 2. Chọn UA-DETRAC ở giai đoạn 1 vì: (i) tải công khai, đã có
nhãn định dạng YOLO; (ii) là bộ benchmark quốc tế được dùng rộng rãi để
tham chiếu; (iii) đại diện cho nhóm dữ liệu "dễ" so với giao thông Việt
Nam — giúp thử nghiệm pipeline trước khi chuyển sang dữ liệu khó hơn.

**Bảng 2**. Phân chia UA-DETRAC dùng trong thực nghiệm.

| Split | Số ảnh | Ghi chú |
|---|---|---|
| train | 67 957 | Học tham số |
| val | 14 128 | Chọn epoch tốt nhất, early stopping |
| test | 56 167 | Đánh giá cuối cùng (chưa dùng trong báo cáo này) |
| **Tổng** | **138 252** | |

## 5.2. Siêu tham số huấn luyện

Cả baseline và mô hình cải tiến được huấn luyện với cùng cấu hình để đảm bảo
so sánh công bằng (Bảng 3).

**Bảng 3**. Siêu tham số huấn luyện.

| Siêu tham số | Giá trị |
|---|---|
| Kích thước ảnh | 640 × 640 |
| Batch size | 16 |
| Số epoch tối đa | 50 |
| Patience (early stopping) | 15 |
| Optimizer | Auto (Ultralytics chọn MuSGD, lr=0.01, momentum=0.9) |
| Weight decay | 5e-4 cho conv weights, 0 cho BN/bias |
| Seed | 42 |
| Pretrained | yolov8n.pt (COCO) |
| Precision | AMP (FP16) |
| Augmentation | Mosaic, MixUp, HSV, flip (mặc định Ultralytics) |
| Số worker DataLoader | 8 |
| Cache | RAM (không thành công do RAM 50GB không đủ 87GB), fallback: không cache train |

## 5.3. Cấu hình phần cứng

- CPU: Intel Core i7-13850HX (16 logical cores khả dụng qua WSL2).
- GPU: NVIDIA RTX 3500 Ada Generation Laptop, 12 GB VRAM.
- RAM: 50 GB (WSL2 đã phân bổ từ host 64 GB).
- OS: Ubuntu 22.04 (WSL2), CUDA 12.x, PyTorch 2.x, Ultralytics 8.4.x.

## 5.4. Chỉ số đánh giá

- **Precision (P)**: tỉ lệ dự đoán đúng trong các dự đoán positive.
- **Recall (R)**: tỉ lệ nhãn dương được phát hiện.
- **mAP@0.5**: mean Average Precision với IoU threshold 0.5, chuẩn PASCAL VOC.
- **mAP@0.5:0.95**: mean Average Precision trung bình theo 10 mức IoU từ 0.5
  đến 0.95, chuẩn COCO — chỉ số nghiêm ngặt hơn.
- **Thời gian huấn luyện**: tổng giờ chạy từ đầu đến khi early stopping.

---

# 6. Kết quả

## 6.1. Kết quả tổng thể

Cả hai mô hình đều hội tụ sớm và dừng theo cơ chế early stopping trước khi
đạt 50 epoch. Baseline hội tụ ở epoch 3, mô hình cải tiến hội tụ ở epoch 5;
cả hai dừng sau 15 epoch liên tiếp không cải thiện. Bảng 4 tổng hợp kết quả
tốt nhất trên tập val (epoch chọn theo mAP@0.5:0.95).

**Bảng 4**. Kết quả tốt nhất trên tập **val** — so sánh baseline và cải tiến.

| Mô hình | Epoch tốt nhất | P | R | mAP@0.5 | mAP@0.5:0.95 | Params | Thời gian train |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n baseline (early stop) | 3 / 18 | 0.799 | 0.731 | **0.816** | **0.622** | 3.00 M | 2h 37m |
| YOLOv8n + CBAM + CW (full 50 ep) | 9 / 50 | 0.824 | 0.681 | 0.784 | 0.610 | 3.10 M | ~5h 00m |
| Δ (cải tiến − baseline) | | +0.025 | −0.050 | **−0.032** | **−0.012** | +0.10 M | +2h 23m |

**Nhận xét chính**: mô hình cải tiến đạt precision cao hơn (+0.025) nhưng recall
giảm đáng kể (−0.050), dẫn đến mAP@0.5 giảm 0.032 điểm và mAP@0.5:0.95 giảm
0.012 điểm so với baseline. Việc huấn luyện đầy đủ 50 epoch (patience = 100 để
loại bỏ cơ chế early stopping) xác nhận rằng cải tiến **không có upside** trên
UA-DETRAC — đường cong mAP đã đạt đỉnh ở epoch 9 và bắt đầu suy giảm nhẹ, dấu
hiệu của slight overfitting trên val split.

**Bảng 4b**. Kết quả trên tập **test** (56 167 ảnh chưa từng thấy trong huấn luyện).

| Mô hình | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| YOLOv8n baseline | 0.6100 | **0.5708** | **0.5948** | **0.4317** |
| YOLOv8n + CBAM + CW | **0.6184** | 0.5407 | 0.5522 | 0.4094 |
| Δ (cải tiến − baseline) | +0.0084 | −0.0301 | **−0.0426** | **−0.0223** |

Kết quả trên test **nhất quán** với val: baseline tốt hơn cải tiến trên cả hai
mAP, cải tiến chỉ vượt về precision nhờ tính "cẩn trọng" của class-weighted BCE.
Khoảng cách trên test còn lớn hơn val (Δ mAP@0.5 = −0.043 vs −0.032 trên val),
gợi ý rằng cải tiến **generalize kém hơn** baseline trên phân phối test.

## 6.2. Đường cong huấn luyện

Bảng 5 và 6 trình bày mAP@0.5:0.95 qua các epoch. Cả hai mô hình đạt đỉnh
rất sớm rồi dao động quanh giá trị đỉnh mà không vượt qua được — dấu hiệu
điển hình của bộ dữ liệu tương đối cân bằng và pretrained COCO đã "biết
sẵn" các lớp phương tiện.

**Bảng 5**. mAP@0.5:0.95 theo epoch — baseline.

| Epoch | mAP@0.5:0.95 | | Epoch | mAP@0.5:0.95 |
|---:|---:|---|---:|---:|
| 1 | 0.586 | | 10 | 0.570 |
| 2 | 0.607 | | 11 | 0.562 |
| **3** | **0.622** ← peak | | 12 | 0.568 |
| 4 | 0.601 | | 13 | 0.569 |
| 5 | 0.602 | | 14 | 0.571 |
| 6 | 0.585 | | 15 | 0.572 |
| 7 | 0.582 | | 16 | 0.572 |
| 8 | 0.582 | | 17 | 0.571 |
| 9 | 0.577 | | 18 | 0.571 |

**Bảng 6**. mAP@0.5:0.95 theo epoch — mô hình cải tiến (full 50 epoch).

| Epoch | mAP@0.5:0.95 | | Epoch | mAP@0.5:0.95 | | Epoch | mAP@0.5:0.95 |
|---:|---:|---|---:|---:|---|---:|---:|
| 1 | 0.498 | | 18 | 0.593 | | 35 | 0.588 |
| 2 | 0.565 | | 19 | 0.594 | | 36 | 0.585 |
| 3 | 0.583 | | 20 | 0.596 | | 37 | 0.583 |
| 5 | 0.601 | | 22 | 0.601 | | 40 | 0.582 |
| 7 | 0.605 | | 25 | 0.608 | | 43 | 0.586 |
| **9** | **0.610** ← peak | | 27 | 0.605 | | 45 | 0.584 |
| 11 | 0.606 | | 29 | 0.599 | | 47 | 0.583 |
| 13 | 0.604 | | 31 | 0.597 | | 48 | 0.584 |
| 15 | 0.598 | | 33 | 0.591 | | 49 | 0.582 |
| 17 | 0.595 | | 34 | 0.590 | | 50 | 0.579 |

Model đạt peak ở epoch 9 (0.610), duy trì cao trong epoch 5-27, sau đó dao động
nhẹ và có xu hướng giảm dần từ epoch 30 trở đi (dấu hiệu slight overfitting).

## 6.3. So sánh precision – recall

Bảng 7 cho thấy sự đánh đổi precision – recall giữa hai mô hình tại epoch
tốt nhất của từng mô hình.

**Bảng 7**. So sánh precision – recall.

| Mô hình | Precision | Recall | F1 (xấp xỉ) |
|---|---:|---:|---:|
| Baseline val (ep. 3) | 0.799 | 0.731 | 0.763 |
| Cải tiến val (ep. 9) | 0.824 | 0.681 | 0.746 |
| Baseline test | 0.610 | 0.571 | 0.590 |
| Cải tiến test | 0.618 | 0.541 | 0.577 |

Trên cả val và test, mô hình cải tiến "cẩn trọng" hơn — dự đoán ít hơn
(recall giảm) nhưng chính xác hơn (precision tăng). Đây là hệ quả điển hình
của class-weighted BCE làm loss chú ý đến các lớp hiếm và khiến model dè dặt
hơn với dự đoán confidence thấp. Trên bộ dữ liệu cân bằng như UA-DETRAC, sự
dè dặt này không đem lại lợi ích, thậm chí làm giảm F1 nhẹ.

---

# 7. Thảo luận

## 7.1. Tại sao cải tiến chưa thắng baseline trên UA-DETRAC?

Có hai giả thuyết bổ sung cho nhau:

**Giả thuyết A — UA-DETRAC vốn "dễ" cho baseline**. UA-DETRAC chỉ có 4 lớp
phương tiện có ranh giới lớp rõ (car, van, bus, others), mật độ vật thể thấp,
điều kiện ánh sáng tốt (chủ yếu ban ngày). Chỉ với pretrained COCO, YOLOv8n
đã đạt mAP@0.5 = 0.816 chỉ sau 3 epoch fine-tune. Trong tình huống này, việc
thêm CBAM (một module chú ý dành cho small objects, occlusion) không tìm được
"đất diễn" — không có tín hiệu attention nào chưa được backbone CSPDarknet
xử lý tốt.

**Giả thuyết B — Class-weighted BCE có thể phản tác dụng khi lớp đã cân bằng**.
Trọng số $w_i = 1/n_i$ tăng cường loss cho lớp hiếm. Khi bốn lớp phương tiện
tương đối cân bằng như trên UA-DETRAC, việc "kéo" model chú ý quá mức đến
những lớp cân bằng-nhẹ có thể làm sai lệch phân phối dự đoán — dẫn đến recall
giảm nhẹ và mAP@0.5 giảm 0.015.

Hai giả thuyết nhất quán với dữ liệu quan sát: precision tăng (0.799 → 0.811)
vì class weight khiến model "cẩn trọng" hơn với các dự đoán không chắc chắn,
nhưng recall giảm (0.731 → 0.721) vì bỏ sót các đối tượng lớp phổ biến (car)
mà nếu không dùng class weight, model đã tự tin dự đoán.

## 7.2. Ý nghĩa cho giai đoạn 2 (dữ liệu Việt Nam)

Kết quả trên UA-DETRAC **không phủ nhận** giá trị của CBAM + class-weighted
BCE — chỉ chỉ ra rằng bối cảnh áp dụng phải phù hợp. Cả hai kỹ thuật đều
được thiết kế cho **dữ liệu khó** với đặc điểm:

- Mật độ vật thể cao, che khuất nhiều → cần attention để tách bối cảnh khỏi
  vật thể (CBAM phát huy).
- Mất cân bằng lớp nặng → cần class-weighted loss để lớp hiếm không bị
  "nuốt" trong gradient (đúng thiết kế).

Dữ liệu UIT-VinaDeveS22 (Trịnh & Nguyễn, 2022) có mật độ cao và mất cân
bằng 40:1 giữa xe máy và xe đạp — đây là bối cảnh nơi cải tiến kỳ vọng phát
huy tác dụng. Giai đoạn 2 sẽ lặp lại thí nghiệm này trên dữ liệu Việt Nam
để kiểm chứng.

## 7.3. Hạn chế của báo cáo

1. **Chưa có accuracy per-class**: chỉ số tổng thể (P, R, mAP) không cho biết
   cải tiến ảnh hưởng đến lớp nào cụ thể. Đây là chỉ số quan trọng cho luận
   điểm về "cải thiện lớp hiếm" và sẽ được bổ sung ở giai đoạn 2.
2. **Ablation CBAM-only đang tiến hành**: chưa tách được đóng góp riêng của
   CBAM và class-weighted BCE. Run `--no-class-weights` đang chạy để cô lập
   tác động CBAM và sẽ được cập nhật vào bản báo cáo tiếp theo.
3. **Chưa đo tốc độ suy luận (FPS)**: yếu tố quan trọng cho ứng dụng thực tế,
   sẽ đo bằng script `evaluate.py` với video mẫu ở giai đoạn tiếp theo.
4. **Baseline dùng patience = 15 (early stop ở epoch 18)**, còn improved dùng
   patience = 100 (train hết 50 epoch). Tuy có chênh lệch về số epoch dừng,
   cả hai mô hình đều đã đạt đỉnh và plateau ổn định trong ít nhất 15 epoch
   sau đó, đảm bảo so sánh có ý nghĩa.
5. **Cấu hình phần cứng bị giới hạn bởi RAM**: cache RAM không hoạt động
   (thiếu 37 GB so với yêu cầu 87 GB) nên phải dùng cache disk. Với batch = 32,
   workers = 16, RTX 3500 Ada 12 GB, epoch time đạt ~380 s/epoch trong run
   full 50 epoch — cải thiện đáng kể so với 600 s/epoch ở run ban đầu.

---

# 8. Kết luận và hướng phát triển

## 8.1. Kết luận

Chúng tôi đã xây dựng pipeline huấn luyện – so sánh cho hai biến thể YOLOv8n
(baseline và cải tiến CBAM + class-weighted BCE) và thực nghiệm trên UA-DETRAC
với cùng cấu hình. Kết quả **trên cả val và test** cho thấy trên bộ dữ liệu
tương đối cân bằng này, cải tiến **thua baseline** về mAP tổng thể:

- **Trên val**: mAP@0.5 giảm 0.032, mAP@0.5:0.95 giảm 0.012.
- **Trên test (56 167 ảnh)**: mAP@0.5 giảm 0.043, mAP@0.5:0.95 giảm 0.022.

Việc huấn luyện cải tiến đầy đủ 50 epoch (loại bỏ early stopping) xác nhận
model đã hội tụ ở epoch 9 và không thể vượt baseline dù cho thêm thời gian.
Kết quả này nhất quán với đặc điểm của UA-DETRAC (cân bằng lớp, mật độ thấp)
và không phủ nhận giá trị lý thuyết của các cải tiến — đúng hơn, nó chỉ ra
rằng bối cảnh áp dụng phải phù hợp với thiết kế của kỹ thuật. Chi tiết lý
luận đã trình bày ở §7.1–7.2.

## 8.2. Việc cần làm ngay

1. **Chạy lại huấn luyện với patience cao hơn** (đã cấu hình patience = 100
   tương đương tắt early stopping) và cấu hình tối ưu (batch = 32,
   workers = 16, cache = disk) để confirm hai điểm: (a) mô hình cải tiến
   có tận dụng thêm epoch để vượt baseline hay không, và (b) đạt bao nhiêu
   mAP tối đa trên UA-DETRAC. Chạy này đang tiến hành, epoch time ước tính
   ~350 s (giảm ~40 % so với 600 s trước tối ưu).
2. **Đánh giá trên tập test** UA-DETRAC (56 167 ảnh) sau khi có `best.pt`
   ổn định.
3. **Chạy ablation**: mô hình cải tiến chỉ CBAM (tắt class weights) để tách
   đóng góp của từng cải tiến.

## 8.3. Hướng phát triển giai đoạn 2

1. **Chuyển sang dữ liệu Việt Nam**: kết hợp Roboflow "Vietnamese vehicle"
   (1 547 ảnh), VisDrone và dữ liệu tự thu thập; xin dữ liệu UIT-VinaDeveS22
   cho tham chiếu học thuật.
2. **Đánh giá per-class + theo điều kiện môi trường** (ngày/đêm/mưa) — đây
   là chỉ số then chốt để chứng minh khoảng trống #3 và #4.
3. **So sánh mở rộng**: thêm YOLOv8s và YOLOv11n làm cột tham chiếu.
4. **Nếu class-weighted BCE thuần không đủ**: chuyển sang Focal Loss
   ($\gamma = 1.5$, $\alpha = 0.25$) đã triển khai sẵn trong pipeline.

---

# Tài liệu tham khảo

1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8*.
   github.com/ultralytics/ultralytics.

2. Wen, L., Du, D., Cai, Z., Lei, Z., Chang, M. C., Qi, H., ... & Lyu, S. (2020).
   UA-DETRAC: A new benchmark and protocol for multi-object detection and
   tracking. *Computer Vision and Image Understanding*, 193, 102907.

3. Tao, Y. (2026). ContextECA2.0 with adaptive window attention for
   YOLOv8n on drone-view traffic. *Informatica*.

4. Khalili, B., & Smyth, A. W. (2024). SOD-YOLOv8 — Enhancing YOLOv8 for
   small object detection in traffic scenes with GFPN, EMA attention and
   Powerful-IoU loss. *ArXiv preprint*.

5. Trịnh, T. Đ., & Nguyễn, T. K. (2022). UIT-VinaDeveS22: A Vietnamese CCTV
   vehicle dataset and comparison of object detection models. *CTU Journal
   of Science*.

6. Nguyễn, V. A., & Phạm, T. H. (2025). Vehicle detection and tracking with
   YOLOv8 + StrongSORT for Vietnamese urban traffic. *Springer ICTA*.

7. Nguyễn, T. B. et al. (2024). A comparative study on Vietnamese traffic
   sign recognition (VTSDB46). *ACM ICIIT*.

8. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional
   Block Attention Module. *ECCV 2018*, 3–19.

9. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal
   loss for dense object detection. *ICCV 2017*, 2980–2988.

---

# Phụ lục A. Cấu trúc thư mục dự án

```
yolov8_finetune/
├── configs/class_mapping.json
├── data/{detrac,data}.yaml
├── models/yolov8_improved/yolov8n-cbam-detrac.yaml
├── scripts/
│   ├── prepare_data.py
│   ├── train_yolo_baseline.py
│   ├── train_yolo_improved.py
│   ├── evaluate.py
│   ├── demo.py
│   └── train.sh
├── notebooks/yolov8_finetune_pipeline.ipynb
├── runs/{yolo_baseline,yolo_improved}/exp/
└── docs/{README, GHI_CHU_LAM_VIEC, BAO_CAO_KET_QUA}.{md,docx}
```

# Phụ lục B. Lệnh chạy thực nghiệm

```bash
# Train baseline
./scripts/train.sh baseline bg

# Train improved
./scripts/train.sh improved bg

# Resume improved từ last.pt
./scripts/train.sh resume bg

# Xem log run mới nhất
tail -f logs/$(ls -t logs/ | head -1)

# Đánh giá + so sánh 2 model
python scripts/evaluate.py
```
