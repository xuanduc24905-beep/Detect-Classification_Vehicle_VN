"""Demo: chạy model tốt nhất trên ảnh/video, vẽ bbox + đếm phương tiện.

Ví dụ:
    python scripts/demo.py --weights runs/yolo_improved/exp/weights/best.pt \
                           --source samples/traffic.mp4 --out results/demo/out.mp4
    python scripts/demo.py --weights ... --source samples/*.jpg --out results/demo/
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_yolo_improved import register_cbam, NeckCBAM  # noqa: E402
sys.modules["__main__"].NeckCBAM = NeckCBAM  # để unpickle checkpoint improved tìm được class

CLASSES = ["motorcycle", "car", "bus", "truck", "bicycle"]
COLORS = [(0, 200, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]


def draw(frame, boxes, cls_ids, confs, counter: Counter | None = None) -> None:
    for (x1, y1, x2, y2), c, cf in zip(boxes, cls_ids, confs):
        c = int(c)
        color = COLORS[c % len(COLORS)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{CLASSES[c]} {cf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if counter is not None:
        h = 24
        y = 24
        for cls_id, name in enumerate(CLASSES):
            n = counter.get(cls_id, 0)
            cv2.putText(frame, f"{name}: {n}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[cls_id], 2, cv2.LINE_AA)
            y += h


def process_image(model, path: Path, out_path: Path, conf: float) -> Counter:
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Không đọc được: {path}")
        return Counter()
    res = model.predict(img, verbose=False, conf=conf, imgsz=640)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    clsi = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    counter = Counter(clsi.tolist())
    draw(img, boxes, clsi, confs, counter)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return counter


def process_video(model, path: Path, out_path: Path, conf: float) -> Counter:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    peak_counter: Counter = Counter()  # max đồng thời qua các frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, verbose=False, conf=conf, imgsz=640)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        clsi = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        cur = Counter(clsi.tolist())
        for k, v in cur.items():
            peak_counter[k] = max(peak_counter[k], v)
        draw(frame, boxes, clsi, confs, cur)
        writer.write(frame)
    cap.release(); writer.release()
    return peak_counter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True,
                   help="Ảnh, video (mp4/avi), hoặc thư mục ảnh")
    p.add_argument("--out", type=Path, default=Path("results/demo/output"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--needs-cbam", action="store_true",
                   help="Bật khi weight là YOLOv8 cải tiến (đăng ký NeckCBAM)")
    args = p.parse_args()

    if args.needs_cbam:
        register_cbam()
    model = YOLO(str(args.weights))

    if args.source.is_dir():
        args.out.mkdir(parents=True, exist_ok=True)
        total: Counter = Counter()
        for img_path in sorted(args.source.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            c = process_image(model, img_path, args.out / img_path.name, args.conf)
            total.update(c)
        counter = total
    elif args.source.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
        out = args.out if args.out.suffix else args.out / (args.source.stem + "_out.mp4")
        counter = process_video(model, args.source, out, args.conf)
    else:
        out = args.out if args.out.suffix else args.out / args.source.name
        counter = process_image(model, args.source, out, args.conf)

    print("\n=== Thống kê phương tiện ===")
    total = sum(counter.values())
    for cls_id, name in enumerate(CLASSES):
        print(f"  {name:12s}: {counter.get(cls_id, 0)}")
    print(f"  {'TOTAL':12s}: {total}")


if __name__ == "__main__":
    main()
