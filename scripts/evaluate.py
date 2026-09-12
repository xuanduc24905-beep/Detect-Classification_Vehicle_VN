"""So sánh 2 model YOLOv8: baseline vs cải tiến (CBAM + class-weighted/focal).

Bảng kết quả (results/tables/comparison.csv) gồm:
  - mAP@0.5, mAP@0.5:0.95
  - precision / recall per-class
  - accuracy per-class (recall/hit-rate theo IoU threshold)
  - FPS đo trên video mẫu (--fps-video)

Cũng xuất biểu đồ so sánh:
  - results/figures/map50_bar.png, map5095_bar.png
  - results/figures/per_class_accuracy.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.train_yolo_improved import register_cbam  # noqa: E402

CLASSES = ["motorcycle", "car", "bus", "truck", "bicycle"]


def eval_yolo(weights: Path, data_yaml: Path, name: str, project: str = "runs/eval") -> dict:
    """Chạy YOLO.val trên split test, trả về dict metrics tổng + per-class."""
    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), split="test", project=project, name=name, plots=True)
    per_cls_p = m.box.p.tolist() if hasattr(m.box, "p") else []
    per_cls_r = m.box.r.tolist() if hasattr(m.box, "r") else []
    per_cls_map50 = m.box.ap50.tolist() if hasattr(m.box, "ap50") else []
    return {
        "map50": float(m.box.map50),
        "map": float(m.box.map),
        "precision": per_cls_p,
        "recall": per_cls_r,
        "map50_per_class": per_cls_map50,
    }


def eval_yolo_accuracy_per_class(weights: Path, data_root: Path,
                                  iou_thr: float = 0.5, conf: float = 0.25) -> np.ndarray:
    """Accuracy per-class: tỉ lệ GT box được match đúng lớp bởi 1 prediction có IoU>=thr.

    Đây là "recall theo lớp" — thuật ngữ 'accuracy per-class' trong bài toán
    detection thường được hiểu là recall/hit-rate per-class.
    """
    model = YOLO(str(weights))
    hits = np.zeros(len(CLASSES), dtype=np.int64)
    total = np.zeros(len(CLASSES), dtype=np.int64)
    img_dir = data_root / "images" / "test"
    lbl_dir = data_root / "labels" / "test"
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = lbl_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = []  # (cls, x1,y1,x2,y2)
        with open(lbl) as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 5:
                    continue
                cls = int(p[0]); cx, cy, bw, bh = map(float, p[1:])
                x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
                gts.append((cls, x1, y1, x2, y2))
                total[cls] += 1
        if not gts:
            continue
        res = model.predict(img, verbose=False, conf=conf, imgsz=640)[0]
        preds = []
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
            preds.append((int(c), *b.tolist()))
        # Greedy match GT -> best-IoU pred
        for cls_gt, gx1, gy1, gx2, gy2 in gts:
            best_iou = 0.0; best_cls = -1
            for cls_p, px1, py1, px2, py2 in preds:
                ix1, iy1 = max(gx1, px1), max(gy1, py1)
                ix2, iy2 = min(gx2, px2), min(gy2, py2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                if inter == 0:
                    continue
                union = (gx2 - gx1) * (gy2 - gy1) + (px2 - px1) * (py2 - py1) - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou; best_cls = cls_p
            if best_iou >= iou_thr and best_cls == cls_gt:
                hits[cls_gt] += 1
    acc = hits / np.maximum(total, 1)
    return acc


def measure_fps(weights: Path, video: Path, warmup: int = 10, max_frames: int = 200) -> float:
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return float("nan")
    for _ in range(warmup):
        ok, frame = cap.read()
        if not ok:
            break
        _ = model.predict(frame, verbose=False, imgsz=640)
    n = 0
    t0 = time.time()
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        _ = model.predict(frame, verbose=False, imgsz=640)
        n += 1
    dt = time.time() - t0
    cap.release()
    return n / dt if dt > 0 else float("nan")


def bar_chart(labels: list[str], values: list[float], title: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel); ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150)
    plt.close(fig)


def grouped_bar(per_class: dict[str, list[float]], out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(CLASSES))
    n = len(per_class)
    w = 0.8 / n
    for i, (name, vals) in enumerate(per_class.items()):
        ax.bar(x + (i - n / 2 + 0.5) * w, vals, w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=15)
    ax.set_ylabel("accuracy / recall (test)"); ax.set_title(title)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yolo-baseline", type=Path,
                   default=Path("runs/yolo_baseline/exp/weights/best.pt"))
    p.add_argument("--yolo-improved", type=Path,
                   default=Path("runs/yolo_improved/exp/weights/best.pt"))
    p.add_argument("--data-yaml", type=Path, default=Path("data/data.yaml"))
    p.add_argument("--data-root", type=Path, default=Path("data/processed"))
    p.add_argument("--fps-video", type=Path, default=None,
                   help="Video mẫu để đo FPS (nếu bỏ qua sẽ ghi NaN)")
    p.add_argument("--out-csv", type=Path, default=Path("results/tables/comparison.csv"))
    args = p.parse_args()

    register_cbam()

    rows = []

    # ---- YOLO models ----
    for name, w in [("yolov8_baseline", args.yolo_baseline),
                    ("yolov8_improved", args.yolo_improved)]:
        if not w.exists():
            print(f"[skip] {name}: weight không tồn tại ({w})")
            continue
        print(f"\n=== Evaluate {name} ===")
        met = eval_yolo(w, args.data_yaml, name)
        acc_pc = eval_yolo_accuracy_per_class(w, args.data_root)
        fps = measure_fps(w, args.fps_video) if args.fps_video else float("nan")
        rows.append({
            "model": name,
            "mAP50": met["map50"],
            "mAP50-95": met["map"],
            "FPS": fps,
            **{f"P_{c}": v for c, v in zip(CLASSES, met["precision"] or [np.nan] * len(CLASSES))},
            **{f"R_{c}": v for c, v in zip(CLASSES, met["recall"] or [np.nan] * len(CLASSES))},
            **{f"acc_{c}": float(a) for c, a in zip(CLASSES, acc_pc)},
        })

    if not rows:
        raise SystemExit("Không có model nào để đánh giá. Train trước rồi chạy lại.")

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[OK] {args.out_csv}")
    print(df.to_string(index=False))

    # ---- Figures ----
    fig_dir = Path("results/figures")
    yolo_rows = df[df["model"].str.startswith("yolov8")]
    if not yolo_rows.empty:
        bar_chart(yolo_rows["model"].tolist(),
                  yolo_rows["mAP50"].tolist(),
                  "mAP@0.5 on test set", "mAP@0.5",
                  fig_dir / "map50_bar.png")
        bar_chart(yolo_rows["model"].tolist(),
                  yolo_rows["mAP50-95"].tolist(),
                  "mAP@0.5:0.95 on test set", "mAP",
                  fig_dir / "map5095_bar.png")

    per_class = {r["model"]: [r[f"acc_{c}"] for c in CLASSES] for r in rows}
    grouped_bar(per_class, fig_dir / "per_class_accuracy.png",
                "Per-class accuracy / recall (test)")
    print(f"[OK] Figures -> {fig_dir}/")


if __name__ == "__main__":
    main()
