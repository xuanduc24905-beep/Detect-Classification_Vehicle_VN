"""Fine-tune YOLOv8 pretrained (baseline, cấu hình mặc định).

Sử dụng Ultralytics API. Mặc định dùng yolov8n.pt, có thể đổi qua --weights.
Log/weight lưu vào runs/yolo_baseline/<name>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, default="weights/yolov8n.pt",
                   help="yolov8n.pt / yolov8s.pt / ... (Ultralytics tự tải nếu thiếu)")
    p.add_argument("--data", type=Path, default=Path("data/data.yaml"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0", help="GPU id, 'cpu' hoặc '0,1'")
    p.add_argument("--project", type=str, default="runs/yolo_baseline")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        # Augment mặc định của Ultralytics đã hợp lý cho scene giao thông
        seed=42,
        pretrained=True,
    )
    # Evaluate ngay trên test split sau khi train (best.pt tự động dùng)
    metrics = model.val(data=str(args.data), split="test",
                        project=args.project, name=f"{args.name}_test", plots=True)
    print("\n[baseline] mAP50 =", metrics.box.map50, " mAP50-95 =", metrics.box.map)


if __name__ == "__main__":
    main()
