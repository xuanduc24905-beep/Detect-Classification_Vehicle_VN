"""Sinh file README.docx giải thích từ CV cơ bản đến project YOLOv8 fine-tune.

Chạy: python scripts/gen_readme_docx.py
Output: docs/README_YOLOv8_FineTune.docx
"""
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / 'docs' / 'README_YOLOv8_FineTune.docx'
OUT.parent.mkdir(parents=True, exist_ok=True)


def set_style(doc):
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)


def h1(doc, text):
    p = doc.add_heading(text, level=1)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)


def h2(doc, text):
    p = doc.add_heading(text, level=2)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x2E, 0x5C, 0x8A)


def h3(doc, text):
    p = doc.add_heading(text, level=3)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x3E, 0x7C, 0xB0)


def para(doc, text, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(11)
    r.bold = bold
    return p


def bullet(doc, text):
    p = doc.add_paragraph(text, style='List Bullet')
    return p


def code(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = 'Consolas'
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    # shading nhẹ
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), 'F1F1F1')
    p.paragraph_format.left_indent = Cm(0.5)
    p._p.get_or_add_pPr().append(shd)
    return p


def note(doc, text):
    p = doc.add_paragraph()
    r = p.add_run('💡 ' + text)
    r.italic = True
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def divider(doc):
    doc.add_paragraph('_' * 60).alignment = WD_ALIGN_PARAGRAPH.CENTER


doc = Document()
set_style(doc)

# ============================================================
# COVER
# ============================================================
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run('HƯỚNG DẪN FINE-TUNE YOLOv8')
r.bold = True
r.font.size = Pt(24)
r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run('Từ thị giác máy tính cơ bản đến project phát hiện phương tiện')
r.font.size = Pt(14)
r.italic = True

doc.add_paragraph()
para(doc, 'Tài liệu dành cho người MỚI HOÀN TOÀN với deep learning và computer vision. '
     'Đọc tuần tự từ chương 1. Mỗi khái niệm được giải thích trước khi dùng.')
doc.add_page_break()

# ============================================================
# MỤC LỤC
# ============================================================
h1(doc, 'Mục lục')
bullet(doc, 'Chương 1 — Thị giác máy tính là gì?')
bullet(doc, 'Chương 2 — Ba bài toán chính: Classification, Detection, Segmentation')
bullet(doc, 'Chương 3 — Mạng nơ-ron tích chập (CNN) — nền tảng')
bullet(doc, 'Chương 4 — YOLO là gì? Vì sao lại "You Only Look Once"?')
bullet(doc, 'Chương 5 — Kiến trúc YOLOv8: Backbone, Neck, Head')
bullet(doc, 'Chương 6 — Dataset đến từ đâu? (Đây KHÔNG phải model tự học)')
bullet(doc, 'Chương 7 — Format nhãn YOLO (.txt) — đọc hiểu từng số')
bullet(doc, 'Chương 8 — Fine-tuning và Transfer Learning')
bullet(doc, 'Chương 9 — Loss function: model học bằng cách nào')
bullet(doc, 'Chương 10 — Đánh giá: Precision, Recall, IoU, mAP')
bullet(doc, 'Chương 11 — Class imbalance và cách xử lý')
bullet(doc, 'Chương 12 — Attention mechanism và CBAM (phần cải tiến)')
bullet(doc, 'Chương 13 — Chi tiết project này')
bullet(doc, 'Chương 14 — Hướng dẫn chạy notebook')
bullet(doc, 'Chương 15 — FAQ (câu hỏi thường gặp)')
doc.add_page_break()

# ============================================================
# CHƯƠNG 1
# ============================================================
h1(doc, 'Chương 1 — Thị giác máy tính (Computer Vision) là gì?')

para(doc, 'Thị giác máy tính (CV) là ngành nghiên cứu cách máy tính "nhìn" và "hiểu" ảnh/video. '
     'Với con người, nhìn ảnh và biết đó là xe hơi là chuyện tự nhiên. Với máy tính, ảnh chỉ là một '
     'ma trận số — mỗi pixel là 3 con số (Red, Green, Blue) từ 0 đến 255.')

para(doc, 'Ví dụ: ảnh 640×480 RGB → tensor shape (3, 480, 640) với 3×480×640 = 921,600 con số. '
     'Nhiệm vụ của CV: biến 921,600 con số này thành thông tin có nghĩa — "trong ảnh có 3 xe hơi, '
     '1 xe buýt, chúng đang di chuyển sang phải, tốc độ ước tính X km/h".')

h2(doc, '1.1. Các cấp độ hiểu ảnh')
bullet(doc, 'Cấp thấp: cạnh, góc, texture, màu chủ đạo')
bullet(doc, 'Cấp trung: object có mặt trong ảnh, vị trí, kích thước')
bullet(doc, 'Cấp cao: hành động đang diễn ra, ý nghĩa cảnh')

para(doc, 'Trước 2012, người ta viết THỦ CÔNG các bộ trích đặc trưng (SIFT, HOG, Haar...). '
     'Rất tốn công và độ chính xác giới hạn. Sau AlexNet 2012, deep learning thống trị — '
     'model tự học đặc trưng từ dữ liệu.')

divider(doc)

# ============================================================
# CHƯƠNG 2
# ============================================================
h1(doc, 'Chương 2 — Ba bài toán chính trong CV')

h2(doc, '2.1. Image Classification (phân loại ảnh)')
para(doc, 'Input: 1 ảnh. Output: 1 nhãn (label) — "đây là chó" hoặc "đây là mèo".')
para(doc, 'Model chỉ trả lời: cả ảnh này chứa gì. KHÔNG biết vị trí object nằm đâu, có bao nhiêu object.')
para(doc, 'Dataset kinh điển: ImageNet (1000 class, 1.2 triệu ảnh).')

h2(doc, '2.2. Object Detection (phát hiện object) — CHỦ ĐỀ PROJECT NÀY')
para(doc, 'Input: 1 ảnh. Output: DANH SÁCH các object, mỗi object gồm:')
bullet(doc, 'Bounding box (khung chữ nhật) — tọa độ (x, y, width, height)')
bullet(doc, 'Class label — loại (car, bus, person...)')
bullet(doc, 'Confidence score — độ tự tin (0-1)')

para(doc, 'Ví dụ output: [(car, box=(120,80,200,150), conf=0.92), (bus, box=(400,50,300,250), conf=0.87), ...]')
para(doc, 'YOLO là một dòng model làm bài toán này.')

h2(doc, '2.3. Semantic / Instance Segmentation (phân đoạn)')
para(doc, 'Không chỉ box mà PIXEL nào thuộc object nào. Cấp độ khó hơn detection.')
para(doc, 'Ví dụ: xe tự lái cần biết pixel nào là đường, pixel nào là vỉa hè, người, xe khác.')

note(doc, 'Project này giải bài toán 2.2 — Object Detection. Model output box + class cho từng xe.')

divider(doc)

# ============================================================
# CHƯƠNG 3
# ============================================================
h1(doc, 'Chương 3 — Convolutional Neural Network (CNN)')

para(doc, 'CNN là "não" của mọi model CV hiện đại. Nó gồm nhiều lớp (layer) xếp chồng, '
     'mỗi lớp biến ảnh thành một biểu diễn abstract hơn.')

h2(doc, '3.1. Phép tích chập (Convolution)')
para(doc, 'Tưởng tượng có 1 filter nhỏ (ví dụ 3×3) trượt qua toàn bộ ảnh. Tại mỗi vị trí, filter '
     'nhân với vùng ảnh dưới nó rồi cộng lại → ra 1 con số. Kết quả là 1 "feature map" mới.')

para(doc, 'Ý nghĩa: mỗi filter có thể học ra một pattern — ví dụ filter chuyên phát hiện cạnh dọc, '
     'filter khác phát hiện cạnh chéo, filter khác phát hiện màu vàng. Nhiều filter → nhiều feature map.')

h2(doc, '3.2. Pooling')
para(doc, 'Thu nhỏ feature map (thường 2×2 → 1) để giảm tính toán và giữ pattern quan trọng.')

h2(doc, '3.3. Xếp chồng nhiều layer')
para(doc, 'Layer đầu học đặc trưng thấp (cạnh, màu). Layer giữa học đặc trưng trung bình (bánh xe, cửa sổ). '
     'Layer cuối học đặc trưng cao cấp (nguyên chiếc xe hơi). Đây gọi là feature hierarchy.')

note(doc, 'Bạn không cần code CNN từ đầu — Ultralytics đã bọc sẵn tất cả trong YOLOv8. '
     'Nhưng hiểu nó giúp bạn biết vì sao model cần nhiều data, vì sao GPU nhanh hơn CPU rất nhiều.')

divider(doc)

# ============================================================
# CHƯƠNG 4
# ============================================================
h1(doc, 'Chương 4 — YOLO: You Only Look Once')

para(doc, 'Trước YOLO (2016), các model detection (R-CNN, Fast R-CNN...) hoạt động 2 giai đoạn:')
bullet(doc, 'Bước 1: đề xuất hàng ngàn "region" nghi ngờ có object')
bullet(doc, 'Bước 2: phân loại từng region')
para(doc, 'Kết quả: chính xác nhưng CHẬM. Không real-time.')

para(doc, 'YOLO đưa ra ý tưởng: chia ảnh thành lưới ô, mỗi ô dự đoán TRỰC TIẾP box + class. '
     'Chỉ 1 lần forward pass — "You Only Look Once". Kết quả: 30-100+ FPS trên GPU, đủ real-time.')

h2(doc, '4.1. Các phiên bản YOLO')
bullet(doc, 'YOLOv1 (2016) — Redmon et al.')
bullet(doc, 'YOLOv2, v3 — cải tiến anchor box, multi-scale')
bullet(doc, 'YOLOv4, v5 — nhiều tối ưu, v5 do Ultralytics đóng gói')
bullet(doc, 'YOLOv6, v7 — nhánh khác nhau (Meituan, WongKinYiu)')
bullet(doc, 'YOLOv8 (2023) — do Ultralytics phát hành, dùng anchor-free, kiến trúc gọn')
bullet(doc, 'YOLOv9, v10, v11 — mới hơn nhưng v8 vẫn phổ biến vì ổn định + tài liệu nhiều')

para(doc, 'Project này dùng YOLOv8 vì:')
bullet(doc, 'API cực dễ dùng (chỉ cần Python: YOLO("yolov8n.pt").train(...))')
bullet(doc, 'Có sẵn nhiều scale: n (nano), s, m, l, x — chọn theo GPU')
bullet(doc, 'Pretrained weight từ COCO đã có sẵn')

divider(doc)

# ============================================================
# CHƯƠNG 5
# ============================================================
h1(doc, 'Chương 5 — Kiến trúc YOLOv8')

para(doc, 'YOLOv8 gồm 3 khối chính: Backbone → Neck → Head.')

h2(doc, '5.1. Backbone (xương sống)')
para(doc, 'Là 1 CNN sâu, nhiệm vụ trích đặc trưng từ ảnh. YOLOv8 dùng module gọi là C2f '
     '(Cross Stage Partial with 2 convolutions, fused). Backbone giảm dần độ phân giải:')
bullet(doc, 'Input 640×640×3')
bullet(doc, 'Sau vài Conv+C2f → 320×320, 160×160, 80×80, 40×40, 20×20')
bullet(doc, 'Feature map 20×20 chứa thông tin cấp cao (nhìn tổng thể), 80×80 chi tiết hơn')

h2(doc, '5.2. Neck (cổ)')
para(doc, 'Nhiệm vụ: pha trộn đặc trưng ở nhiều scale khác nhau. Object nhỏ (xe đạp xa) cần feature map '
     'độ phân giải cao, object lớn (xe buýt gần) cần feature map thấp. Neck lấy P3 (80×80), P4 (40×40), '
     'P5 (20×20) rồi kết hợp qua Upsample + Concat + C2f.')
para(doc, 'Kiến trúc neck của YOLOv8 gọi là PANet (Path Aggregation Network).')

h2(doc, '5.3. Head (đầu ra)')
para(doc, 'Ba nhánh Detect, mỗi nhánh dự đoán trên 1 scale (P3/P4/P5). Mỗi vị trí lưới dự đoán:')
bullet(doc, '4 giá trị offset (chỉnh box)')
bullet(doc, 'nc giá trị class score (nc = số class)')
para(doc, 'YOLOv8 là "anchor-free" — không cần định nghĩa trước các box mẫu (anchor) như v3-v5.')

h2(doc, '5.4. Các scale')
para(doc, 'Ultralytics phát hành 5 scale của YOLOv8:')
bullet(doc, 'YOLOv8n — nano  — 3.2M params, imgsz 640, chạy được cả trên CPU/laptop')
bullet(doc, 'YOLOv8s — small — 11.2M params')
bullet(doc, 'YOLOv8m — medium — 25.9M')
bullet(doc, 'YOLOv8l — large  — 43.7M')
bullet(doc, 'YOLOv8x — xlarge — 68.2M — GPU >=16GB')

para(doc, 'Project này dùng YOLOv8n — nhẹ, đủ tốt cho fine-tune và demo.')

divider(doc)

# ============================================================
# CHƯƠNG 6
# ============================================================
h1(doc, 'Chương 6 — Dataset đến từ đâu? (Model KHÔNG tự học)')

para(doc, 'ĐÂY LÀ HIỂU LẦM QUAN TRỌNG NHẤT. Model deep learning KHÔNG "tự nhìn video rồi tự hiểu". '
     'Nó học từ CẶP (ảnh, nhãn) mà CON NGƯỜI đã dán sẵn.', bold=True)

h2(doc, '6.1. Nhãn (label) được tạo ra như thế nào?')
para(doc, 'Với object detection, mỗi ảnh cần 1 file nhãn liệt kê:')
bullet(doc, 'Có bao nhiêu object trong ảnh')
bullet(doc, 'Mỗi object: tọa độ bounding box + class')

para(doc, 'Cách tạo:')
bullet(doc, 'Cách 1 — thủ công: dùng tool (LabelImg, CVAT, Roboflow) mở ảnh, kéo chuột vẽ box quanh xe, '
     'chọn class từ menu. Với dataset UA-DETRAC ~140,000 frame — vài người ngồi vẽ hàng tháng.')
bullet(doc, 'Cách 2 — bán tự động: dùng model có sẵn dự đoán trước, người review + chỉnh sửa.')
bullet(doc, 'Cách 3 — crowdsource: chia nhỏ cho nhiều người trên Amazon MTurk.')

h2(doc, '6.2. UA-DETRAC — dataset trong project')
para(doc, 'UA-DETRAC là dataset công khai do University at Albany + Chinese Academy of Sciences công bố. '
     'Họ đặt camera ở cầu vượt Bắc Kinh + Thiên Tân, quay video giao thông, rồi thuê người dán nhãn '
     'từng xe trong từng frame — kết quả:')
bullet(doc, '10 giờ video, ~140,000 frame')
bullet(doc, '~1.2 triệu bounding box được vẽ tay')
bullet(doc, '4 class: car, bus, van, others')
bullet(doc, 'Lưu ở format XML riêng (DETRAC-XML)')

h2(doc, '6.3. Nếu tôi có camera riêng thì sao?')
para(doc, 'Bạn KHÔNG thể chỉ cắm video vào rồi model tự học. Phải:')
bullet(doc, 'Cắt video thành frame ảnh (ffmpeg)')
bullet(doc, 'Ngồi vẽ box thủ công cho từng frame — chỗ này tốn thời gian nhất')
bullet(doc, 'Export ra format YOLO (mỗi ảnh 1 file .txt)')
bullet(doc, 'Trộn vào dataset lớn hơn hoặc train riêng')

para(doc, 'Có 1 tricks: dùng model đã train sẵn (như YOLOv8n COCO) để tự động đề xuất box, '
     'rồi bạn chỉ review và chỉnh. Roboflow có sẵn tính năng này.')

note(doc, 'Model học pattern từ label. Label kém = model kém. "Garbage in, garbage out." '
     'Chất lượng label quan trọng hơn số lượng.')

divider(doc)

# ============================================================
# CHƯƠNG 7
# ============================================================
h1(doc, 'Chương 7 — Format YOLO (.txt)')

para(doc, 'YOLO yêu cầu mỗi ảnh có 1 file .txt cùng tên. Ví dụ ảnh img001.jpg → nhãn img001.txt.')

para(doc, 'Mỗi dòng trong .txt = 1 object, format:')
code(doc, 'class_id  x_center  y_center  width  height')

para(doc, 'Trong đó:')
bullet(doc, 'class_id — số nguyên (0, 1, 2, 3) tương ứng thứ tự trong data.yaml')
bullet(doc, 'x_center, y_center — tâm box, CHUẨN HÓA về [0, 1] (chia cho width/height ảnh)')
bullet(doc, 'width, height — kích thước box, cũng chuẩn hóa về [0, 1]')

h2(doc, '7.1. Ví dụ')
para(doc, 'Ảnh 640×480, có 1 xe hơi (class 0) tâm ở (320, 240) box size 100×80:')
code(doc, '0  0.500000  0.500000  0.156250  0.166667')
para(doc, 'Giải thích:')
bullet(doc, 'x_center = 320/640 = 0.5')
bullet(doc, 'y_center = 240/480 = 0.5')
bullet(doc, 'width = 100/640 = 0.15625')
bullet(doc, 'height = 80/480 = 0.16667')

h2(doc, '7.2. File data.yaml')
para(doc, 'Ultralytics đọc file này để biết dataset ở đâu và class là gì:')
code(doc, '''path: /home/user/data/detrac
train: images/train
val:   images/val
test:  images/test

nc: 4
names:
  0: car
  1: bus
  2: van
  3: others''')

divider(doc)

# ============================================================
# CHƯƠNG 8
# ============================================================
h1(doc, 'Chương 8 — Fine-tuning và Transfer Learning')

para(doc, 'Câu hỏi: train YOLOv8 từ đầu (random weight) cần bao nhiêu ảnh?')
para(doc, 'Trả lời: ~100,000+ ảnh, vài trăm giờ GPU. Không khả thi cho project cá nhân.')

para(doc, 'Giải pháp: Transfer Learning.')

h2(doc, '8.1. Ý tưởng')
para(doc, 'Có sẵn 1 model đã train trên COCO (80 class, 118,000 ảnh) → weight của model này '
     'đã "biết nhìn" — nó nhận ra cạnh, texture, hình dạng object nói chung. Ta chỉ cần:')
bullet(doc, 'Giữ nguyên phần lớn weight (backbone)')
bullet(doc, 'Thay head cuối (từ 80 class COCO → 4 class DETRAC)')
bullet(doc, 'Train tiếp trên dataset mới → model "chỉnh" lại cho phù hợp')

para(doc, 'Gọi là FINE-TUNING. Cần 10-100× ít data hơn và thời gian train ngắn hơn nhiều.')

h2(doc, '8.2. Trong code')
code(doc, '''from ultralytics import YOLO

model = YOLO("yolov8n.pt")        # load pretrained COCO weight
model.train(
    data="data/detrac.yaml",       # dataset mới
    epochs=50,                     # số vòng lặp
    imgsz=640,
    batch=16,
)''')

h2(doc, '8.3. Fine-tune với nhiều dataset')
para(doc, 'Câu hỏi: fine-tune xong DETRAC rồi lấy dataset khác train tiếp có được không?')
para(doc, 'Hai cách:', bold=True)

h3(doc, 'Cách A — Merge (khuyến nghị)')
para(doc, 'Gộp tất cả dataset thành 1 pool → train 1 lần trên toàn bộ. Ưu điểm: model được tiếp xúc '
     'với tất cả dữ liệu ở mỗi epoch → không quên gì cả.')

h3(doc, 'Cách B — Sequential (continual learning)')
para(doc, 'Train xong DETRAC → dùng best.pt làm điểm khởi động → train tiếp dataset mới.')
para(doc, 'Rủi ro: "catastrophic forgetting" — model quá tập trung dataset mới, quên DETRAC. '
     'Giảm rủi ro bằng cách: mix 20-30% dữ liệu cũ vào batch mới, hoặc giảm learning rate.')

h3(doc, 'Với 10 dataset')
para(doc, 'Dùng cách A. Nhưng phải thống nhất class trước:')
bullet(doc, 'Dataset A: car, bus, van, others (DETRAC)')
bullet(doc, 'Dataset B: car, truck, motorcycle')
bullet(doc, 'Dataset C: xe hơi, xe khách, xe tải')
para(doc, 'Bạn phải quyết định tập class chung — ví dụ [car, bus, truck, motorcycle, other] — '
     'rồi remap tất cả dataset về tập này (viết script convert). Xong mới gộp.')

divider(doc)

# ============================================================
# CHƯƠNG 9
# ============================================================
h1(doc, 'Chương 9 — Loss function: model học bằng cách nào')

para(doc, 'Loss = "độ sai" giữa dự đoán và ground truth. Train = tìm weight để loss nhỏ nhất.')

para(doc, 'YOLOv8 có 3 thành phần loss:')

h2(doc, '9.1. Box loss (regression)')
para(doc, 'Đo sai lệch tọa độ box dự đoán vs box thật. Dùng CIoU (Complete IoU) — không chỉ đo diện tích '
     'chồng lấn mà còn khoảng cách tâm và tỷ lệ khung.')

h2(doc, '9.2. Cls loss (classification)')
para(doc, 'Đo sai lệch phân loại. Dùng Binary Cross-Entropy — với mỗi class dự đoán "có/không có".')

h2(doc, '9.3. DFL loss (Distribution Focal Loss)')
para(doc, 'Kỹ thuật mới của v8 — thay vì regress 1 giá trị cho mỗi cạnh box, model dự đoán 1 phân phối '
     '→ chính xác hơn ở biên.')

para(doc, 'Tổng loss = w1 · box_loss + w2 · cls_loss + w3 · dfl_loss')
para(doc, 'Ultralytics tự cân bằng w1, w2, w3.')

h2(doc, '9.4. Training loop')
code(doc, '''for epoch in range(epochs):
    for batch in dataloader:
        images, labels = batch
        preds = model(images)               # forward
        loss = compute_loss(preds, labels)  # tính sai số
        loss.backward()                     # backprop — tính gradient
        optimizer.step()                    # cập nhật weight
        optimizer.zero_grad()''')

para(doc, 'Bạn KHÔNG cần viết vòng lặp này. Ultralytics `.train()` bọc sẵn tất cả.')

divider(doc)

# ============================================================
# CHƯƠNG 10
# ============================================================
h1(doc, 'Chương 10 — Đánh giá model')

h2(doc, '10.1. IoU (Intersection over Union)')
para(doc, 'Đo mức chồng lấn giữa box dự đoán và box thật.')
code(doc, 'IoU = diện tích giao / diện tích hợp')
para(doc, 'IoU = 1 → hoàn hảo. IoU = 0 → không chồng chút nào.')
para(doc, 'Ngưỡng phổ biến: IoU >= 0.5 = "true positive" (dự đoán đúng).')

h2(doc, '10.2. Precision, Recall')
bullet(doc, 'Precision = TP / (TP + FP) — trong các box model DỰ ĐOÁN, bao nhiêu % là đúng?')
bullet(doc, 'Recall    = TP / (TP + FN) — trong các box THẬT SỰ có, bao nhiêu % model tìm được?')

para(doc, 'Precision cao + Recall thấp: model quá thận trọng, bỏ sót nhiều.')
para(doc, 'Recall cao + Precision thấp: model quá nhạy, dự đoán bừa.')

h2(doc, '10.3. mAP (mean Average Precision) — metric chính')
para(doc, 'AP = diện tích dưới đường cong Precision-Recall cho 1 class.')
para(doc, 'mAP = trung bình AP của tất cả class.')
bullet(doc, 'mAP@0.5 — IoU threshold 0.5. Đơn giản, dễ đạt cao.')
bullet(doc, 'mAP@0.5:0.95 — trung bình mAP với IoU từ 0.5 → 0.95 (bước 0.05). Khó hơn, khắt khe hơn về vị trí box.')

para(doc, 'Trong project, ta báo cả 2. mAP@0.5:0.95 là metric "chuẩn" trong bài báo học thuật.')

h2(doc, '10.4. Confusion matrix')
para(doc, 'Ma trận cho biết model nhầm class nào với class nào. Ví dụ: van bị nhầm nhiều với car → '
     'có thể do van/car nhìn giống nhau, hoặc dataset ít van.')

divider(doc)

# ============================================================
# CHƯƠNG 11
# ============================================================
h1(doc, 'Chương 11 — Class Imbalance')

para(doc, 'Trong dataset thực tế, các class thường KHÔNG cân bằng. DETRAC ví dụ:')
bullet(doc, 'car: 70-80%')
bullet(doc, 'bus: 5%')
bullet(doc, 'van: 15%')
bullet(doc, 'others: 5%')

para(doc, 'Vấn đề: model học chủ yếu về car (do thấy nhiều nhất), bỏ qua bus/others. '
     'Recall của class hiếm sẽ thấp.')

h2(doc, '11.1. Class-weighted loss')
para(doc, 'Nhân trọng số vào cls loss theo tỉ lệ nghịch với tần suất:')
code(doc, 'weight_c = N_total / (num_classes × N_c)')
para(doc, 'Class hiếm → weight lớn → sai class hiếm bị "phạt" nặng hơn → model buộc phải học.')

h2(doc, '11.2. Focal Loss')
para(doc, 'Ý tưởng: giảm loss của các sample DỄ (model đã dự đoán đúng với confidence cao), '
     'tăng trọng số các sample KHÓ. Công thức:')
code(doc, 'focal = (1 - p)^gamma × BCE')
para(doc, 'gamma thường 1.5-2. Sample nào model đã confident (p cao) → (1-p)^gamma nhỏ → loss giảm.')

h2(doc, '11.3. Data augmentation')
para(doc, 'Nhân bản ảnh của class hiếm qua các phép biến đổi (flip, rotate, color jitter, mosaic). '
     'Ultralytics tự làm mosaic augmentation mặc định.')

divider(doc)

# ============================================================
# CHƯƠNG 12
# ============================================================
h1(doc, 'Chương 12 — Attention & CBAM (phần cải tiến)')

para(doc, 'Attention là ý tưởng cho model "chú ý" nhiều hơn vào vùng ảnh quan trọng, ít hơn vào vùng nhiễu.')

h2(doc, '12.1. CBAM (Convolutional Block Attention Module)')
para(doc, 'CBAM có 2 phần:')
bullet(doc, 'Channel Attention: chỉ ra kênh (channel) feature map nào quan trọng — ví dụ kênh chuyên phát hiện bánh xe > kênh chuyên phát hiện lá cây.')
bullet(doc, 'Spatial Attention: chỉ ra VÙNG NÀO trong feature map quan trọng — ví dụ giữa ảnh > góc ảnh.')

para(doc, 'Cả 2 tạo ra 1 "mask" nhân vào feature map → khuếch đại vùng/kênh quan trọng, giảm vùng nhiễu.')

h2(doc, '12.2. Chèn CBAM vào đâu?')
para(doc, 'Project này chèn CBAM sau 3 output của neck (P3/P4/P5 — trước khi vào head Detect). '
     'Ưu điểm: model tinh chỉnh attention ở đúng scale mà detect head sử dụng.')

para(doc, 'Chi phí: chỉ thêm ~0.1M param cho YOLOv8n (từ 3.2M → 3.3M). Rất nhẹ.')

h2(doc, '12.3. Kết quả kỳ vọng')
para(doc, 'CBAM + class-weighted loss thường cải thiện 1-3% mAP, và ~5-10% recall của class hiếm. '
     'Không phải "phép màu" nhưng đủ để tạo giá trị nghiên cứu.')

divider(doc)

# ============================================================
# CHƯƠNG 13
# ============================================================
h1(doc, 'Chương 13 — Chi tiết project')

h2(doc, '13.1. Cấu trúc thư mục')
code(doc, '''yolov8_finetune/
├── data/
│   ├── raw/ua_detrac/          # dataset gốc (symlink từ kagglehub cache)
│   ├── detrac/                 # sau convert: images/{train,val,test}, labels/{train,val,test}
│   └── detrac.yaml             # config cho Ultralytics
├── models/
│   └── yolov8_improved/
│       └── yolov8n-cbam.yaml   # kiến trúc + CBAM
├── notebooks/
│   └── yolov8_finetune_pipeline.ipynb   # PIPELINE ĐẦY ĐỦ, chạy từ đầu tới cuối
├── scripts/
│   ├── prepare_data.py         # (nếu bạn có dataset dạng khác)
│   ├── train_yolo_baseline.py  # train baseline command-line
│   ├── train_yolo_improved.py  # train CBAM command-line
│   ├── evaluate.py             # so sánh 2 model
│   └── demo.py                 # inference trên ảnh/video
├── runs/                       # output của Ultralytics
│   ├── yolo_baseline/exp/weights/best.pt
│   └── yolo_improved/exp/weights/best.pt
├── weights/yolov8n.pt          # pretrained COCO
└── README.md''')

h2(doc, '13.2. 2 model được huấn luyện')
bullet(doc, 'baseline — YOLOv8n mặc định, fine-tune trực tiếp')
bullet(doc, 'improved — YOLOv8n + CBAM (attention) + class-weighted BCE')

para(doc, 'Mục đích: so sánh xem 2 cải tiến trên có giúp tăng mAP + recall không.')

h2(doc, '13.3. Yêu cầu phần cứng')
para(doc, 'Kịch bản chuẩn (batch=16, imgsz=640):')
bullet(doc, 'GPU >= 8GB VRAM (RTX 3060, 3070, 3500 Ada, 4060+)')
bullet(doc, 'RAM 16GB+')
bullet(doc, 'Disk 20GB+ (dataset ~5GB + cache ~5GB + runs ~2GB)')

para(doc, 'Với GPU 4GB (như RTX 3050 4GB laptop):')
bullet(doc, 'Giảm batch=4 hoặc 8')
bullet(doc, 'Giảm imgsz=512 (thay vì 640)')
bullet(doc, 'Tăng thời gian train ~2-3× — nhưng vẫn chạy được')
bullet(doc, 'YOLOv8n là scale nhỏ nhất — không dùng s/m/l/x')

para(doc, 'CPU-only: chạy được YOLOv8n với batch nhỏ nhưng RẤT chậm (10-20× so với GPU). Chỉ nên dùng để test code, không train thật.')

divider(doc)

# ============================================================
# CHƯƠNG 14
# ============================================================
h1(doc, 'Chương 14 — Hướng dẫn chạy notebook')

h2(doc, '14.1. Chuẩn bị môi trường')
code(doc, '''# Tạo conda env
conda create -n yolov8_ft python=3.10 -y
conda activate yolov8_ft

# PyTorch (CUDA 12.4 cho driver mới)
pip install --index-url https://download.pytorch.org/whl/cu124 \\
    torch==2.5.1 torchvision==0.20.1

# Dependencies
pip install -r requirements.txt
pip install kagglehub jupyter''')

h2(doc, '14.2. Setup Kaggle API')
para(doc, 'Truy cập kaggle.com → Account → Create New API Token → tải kaggle.json về.')
code(doc, '''mkdir -p ~/.config/kaggle
mv ~/Downloads/kaggle.json ~/.config/kaggle/
chmod 600 ~/.config/kaggle/kaggle.json''')

h2(doc, '14.3. Chạy notebook')
code(doc, '''cd yolov8_finetune
jupyter lab
# → mở notebooks/yolov8_finetune_pipeline.ipynb
# → Kernel > Restart & Run All''')

para(doc, 'Notebook sẽ tự:')
bullet(doc, 'Kiểm tra GPU')
bullet(doc, 'Tải UA-DETRAC (~5GB, mất 5-15 phút tùy mạng)')
bullet(doc, 'Parse XML, convert YOLO format (~10-30 phút)')
bullet(doc, 'Train baseline (~2-4 giờ trên RTX 3500, ~6-10 giờ trên RTX 3050 4GB)')
bullet(doc, 'Train improved (~ tương đương)')
bullet(doc, 'Đánh giá + so sánh + visualize')

h2(doc, '14.4. Chạy nhanh (demo mode)')
para(doc, 'Trong cell 0, đổi:')
code(doc, 'QUICK_MODE = True')
para(doc, 'Chỉ dùng 10 sequence, train 5 epoch. Xong sau ~20-30 phút. Chỉ để xem pipeline chạy được, không đủ chất lượng đánh giá thật.')

divider(doc)

# ============================================================
# CHƯƠNG 15
# ============================================================
h1(doc, 'Chương 15 — FAQ')

h3(doc, 'Q1. Model có thể tự học từ video camera không?')
para(doc, 'Không. Model học từ CẶP (ảnh, nhãn) đã được người dán. Bạn phải cắt video thành ảnh + '
     'dùng tool vẽ box thủ công (LabelImg, CVAT, Roboflow) rồi mới train.')

h3(doc, 'Q2. RTX 3050 4GB có chạy được không?')
para(doc, 'Được. Đổi batch=4, imgsz=512. Train sẽ chậm ~2-3× nhưng ổn.')

h3(doc, 'Q3. Push code lên GitHub xong máy khác pull về chạy được không?')
para(doc, 'Được với code. Nhưng:')
bullet(doc, 'Weight best.pt: không push vào git (nặng) — dùng GitHub Releases hoặc HuggingFace')
bullet(doc, 'Dataset ~5GB: không push — người khác tự tải lại từ Kaggle')
bullet(doc, 'Path trong data.yaml: phải chỉnh lại theo máy mới')

h3(doc, 'Q4. Fine-tune xong dataset A, tôi lấy dataset B train tiếp được không?')
para(doc, 'Được nhưng có 2 cách. Khuyến nghị GỘP A+B rồi train 1 lần (tránh model quên A). '
     'Nếu train tuần tự thì mix 20-30% A vào batch B để nhắc nhở model.')

h3(doc, 'Q5. Có 10 dataset thì làm sao?')
para(doc, 'Gộp hết. Nhưng phải REMAP class về 1 tập chung trước — vì mỗi dataset có tập class khác nhau. '
     'Viết script convert từng cái về format YOLO chung, gộp thư mục, train 1 lần.')

h3(doc, 'Q6. mAP bao nhiêu là "tốt"?')
para(doc, 'Phụ thuộc dataset. Với DETRAC (dataset "sạch", ảnh camera cố định), YOLOv8n fine-tune '
     'thường đạt mAP@0.5 = 0.85-0.92. Nếu bạn train ảnh giao thông đường phố Việt Nam (nhiều xe máy, '
     'chồng chéo), mAP@0.5 = 0.6-0.75 đã là tốt.')

h3(doc, 'Q7. Train xong thì file best.pt dùng thế nào?')
para(doc, 'Load bằng Ultralytics rồi predict:')
code(doc, '''from ultralytics import YOLO
model = YOLO("runs/yolo_baseline/exp/weights/best.pt")
results = model("path/to/image.jpg")
results[0].show()''')

h3(doc, 'Q8. Train mất bao lâu?')
para(doc, 'Với DETRAC full (~60,000 frame train), YOLOv8n, batch 16, 50 epoch:')
bullet(doc, 'RTX 3500 Ada 12GB — 2-4 giờ')
bullet(doc, 'RTX 3060 12GB — 3-5 giờ')
bullet(doc, 'RTX 3050 4GB (batch=4) — 6-10 giờ')
bullet(doc, 'CPU — không nên nghĩ đến (30+ giờ)')

h3(doc, 'Q9. Tôi thấy Task Manager Windows báo GPU 0% mà iGPU Intel 67%?')
para(doc, 'Windows Task Manager KHÔNG đọc chính xác GPU compute từ WSL2. Muốn xem đúng, '
     'chạy trong terminal WSL:')
code(doc, 'nvidia-smi')
para(doc, 'Sẽ thấy util và memory used của NVIDIA GPU thật.')

h3(doc, 'Q10. Sau khi train xong nên deploy như thế nào?')
para(doc, 'Nhiều option:')
bullet(doc, 'Export ONNX: model.export(format="onnx") → chạy trên nhiều nền tảng')
bullet(doc, 'Export TensorRT: cho tốc độ cao nhất trên NVIDIA GPU')
bullet(doc, 'FastAPI server: viết endpoint /predict nhận ảnh trả JSON')
bullet(doc, 'Gradio/Streamlit: demo UI nhanh')

divider(doc)

# ============================================================
# KẾT
# ============================================================
h1(doc, 'Lời kết')

para(doc, 'Bạn đã đọc qua toàn bộ hành trình: từ pixel → CNN → YOLO → dataset → training → đánh giá → cải tiến. '
     'Cần nhớ 3 điều quan trọng nhất:')

para(doc, '1. DATA LÀ VUA. Model tốt cần label chất lượng, cân bằng, đủ đa dạng. Đầu tư thời gian '
     'chuẩn bị dữ liệu > đầu tư vào tối ưu hyperparameter.', bold=True)

para(doc, '2. FINE-TUNE > TRAIN FROM SCRATCH. Luôn khởi động từ pretrained weight khi có thể. '
     'Tiết kiệm data + thời gian hàng chục lần.', bold=True)

para(doc, '3. ĐÁNH GIÁ TRÊN VAL/TEST, KHÔNG PHẢI TRAIN. Loss train giảm KHÔNG có nghĩa model tốt — '
     'phải xem mAP trên val/test. Nếu train giảm mà val tăng → overfit.', bold=True)

para(doc, 'Chúc bạn train thành công!', bold=True)

divider(doc)

para(doc, 'Tài liệu tham khảo:')
bullet(doc, 'Ultralytics docs — https://docs.ultralytics.com')
bullet(doc, 'YOLOv8 paper (chưa chính thức, xem bản arXiv của các variant)')
bullet(doc, 'CBAM paper — Woo et al. 2018, arXiv:1807.06521')
bullet(doc, 'UA-DETRAC — Wen et al. 2020, arXiv:1511.04136')
bullet(doc, 'PyTorch tutorial — https://pytorch.org/tutorials')

# Save
doc.save(OUT)
print(f'✓ Đã sinh file: {OUT}')
print(f'  Kích thước: {OUT.stat().st_size / 1024:.1f} KB')
