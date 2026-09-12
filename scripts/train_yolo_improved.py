"""Train YOLOv8 CẢI TIẾN: CBAM ở neck + class-weighted loss.

Hai cải tiến chính so với baseline:
  1) Kiến trúc `models/yolov8_improved/yolov8-cbam.yaml`: chèn module CBAM
     (đã có sẵn trong Ultralytics) sau các C2f cuối cùng của neck, tăng cường
     đặc trưng cho 3 scale detect.
  2) Loss: gán `model.class_weights` = 1/freq (chuẩn hoá) — Ultralytics 8.4.x
     đã hỗ trợ nhân trọng số này vào BCE cls loss. Có option --focal để đổi
     BCE sang FocalLoss (gamma/alpha) qua monkey-patch.

Yêu cầu: chạy `scripts/prepare_data.py` trước để có data/processed/.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import inspect

import torch
import yaml
from torch import nn
from ultralytics import YOLO
from ultralytics.nn import tasks as _tasks
from ultralytics.nn.modules.conv import CBAM
from ultralytics.utils import loss as _loss_mod


class NeckCBAM(nn.Module):
    """CBAM wrapper cho parse_model của Ultralytics.

    Signature (c1, c2, kernel_size=7) khớp cách parse_model gọi các module
    trong `base_modules` (auto-inject c1=ch[f], c2=args[0] đã scale theo width).
    Ta ép c1 == c2 vì CBAM giữ nguyên số kênh.
    """

    def __init__(self, c1: int, c2: int, kernel_size: int = 7):
        super().__init__()
        assert c1 == c2, f"NeckCBAM yêu cầu c1==c2, nhận {c1}!={c2}"
        self.m = CBAM(c1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


def register_cbam() -> None:
    """Đăng ký NeckCBAM vào parse_model + thêm vào base_modules.

    parse_model.base_modules là biến local nên phải patch bằng cách re-exec
    source với NeckCBAM chèn vào frozenset.
    """
    _tasks.NeckCBAM = NeckCBAM
    src = inspect.getsource(_tasks.parse_model)
    if "NeckCBAM," in src:
        return  # đã patch rồi
    patched = src.replace(
        "base_modules = frozenset(\n        {",
        "base_modules = frozenset(\n        {\n            NeckCBAM,",
    )
    ns = {**_tasks.__dict__, "NeckCBAM": NeckCBAM}
    exec(compile(patched, "<patched-parse_model>", "exec"), ns)
    _tasks.parse_model = ns["parse_model"]


def compute_class_weights(labels_dir: Path, num_classes: int) -> torch.Tensor:
    """1/freq chuẩn hoá về mean=1 để không thay đổi độ lớn loss."""
    counter: Counter = Counter()
    for f in labels_dir.glob("*.txt"):
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 5:
                    counter[int(parts[0])] += 1
    counts = torch.tensor([max(counter.get(i, 0), 1) for i in range(num_classes)],
                          dtype=torch.float32)
    w = counts.sum() / (num_classes * counts)
    return w


def patch_focal_loss(gamma: float, alpha: float) -> None:
    """Thay BCE cls trong v8DetectionLoss bằng FocalLoss (giữ nguyên class_weights)."""
    from torch import nn

    orig_init = _loss_mod.v8DetectionLoss.__init__

    def patched_init(self, model, tal_topk=10, tal_topk2=None):
        orig_init(self, model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        class _FocalBCE(nn.Module):
            """Focal BCE giữ shape (bs, na, nc) để tương thích v8DetectionLoss."""

            def forward(_self, pred, target):
                p = pred.sigmoid()
                p_t = target * p + (1 - target) * (1 - p)
                mod = (1.0 - p_t).clamp(min=1e-6) ** gamma
                loss = bce(pred, target) * mod
                if alpha > 0:
                    a = target * alpha + (1 - target) * (1 - alpha)
                    loss = loss * a
                return loss

        self.bce = _FocalBCE()

    _loss_mod.v8DetectionLoss.__init__ = patched_init


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", type=Path,
                   default=Path("models/yolov8_improved/yolov8-cbam.yaml"),
                   help="File .yaml kiến trúc (có thể thêm scale hậu tố, vd yolov8s-cbam.yaml)")
    p.add_argument("--scale", type=str, default="n", choices=list("nsmlx"),
                   help="Scale n/s/m/l/x (Ultralytics dùng suffix trong tên file .yaml)")
    p.add_argument("--pretrained", type=str, default="weights/yolov8n.pt",
                   help="Weight pretrained để khởi tạo backbone (để '' để train from scratch)")
    p.add_argument("--data", type=Path, default=Path("data/data.yaml"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=str, default="runs/yolo_improved")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--resume", action="store_true",
                   help="Resume từ runs/.../weights/last.pt (giữ optimizer/epoch/EMA)")
    p.add_argument("--no-class-weights", action="store_true",
                   help="Tắt class-weighted loss (để so sánh ablation)")
    p.add_argument("--focal", action="store_true",
                   help="Đổi BCE cls sang FocalLoss")
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--focal-alpha", type=float, default=0.25)
    p.add_argument("--cache", type=str, default=None,
                   help="'ram' | 'disk' | None — cache ảnh để tăng tốc epoch")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--exist-ok", action="store_true",
                   help="Ghi đè thư mục run nếu đã tồn tại (giống notebook)")
    args = p.parse_args()

    register_cbam()

    # Resume: bỏ qua build-from-yaml và pretrained, load thẳng last.pt
    if args.resume:
        last = Path(args.project) / args.name / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"Không thấy {last} để resume")
        print(f"[improved] Resume từ {last}")
        model = YOLO(str(last))
        model.train(resume=True)
        metrics = model.val(data=str(args.data), split="test",
                            project=args.project, name=f"{args.name}_test", plots=True)
        print("\n[improved] mAP50 =", metrics.box.map50, " mAP50-95 =", metrics.box.map)
        return

    # Build model từ yaml, load weight pretrained nếu có
    arch_file = args.arch
    # Ultralytics đọc scale từ tên file: yolov8n-cbam.yaml, yolov8s-cbam.yaml...
    # Nếu người dùng để yolov8-cbam.yaml, ta chèn scale vào để nó nhận
    if arch_file.stem == "yolov8-cbam":
        # tạo bản copy với scale suffix bằng cách truyền model= chuỗi
        model_arg = f"yolov8{args.scale}-cbam.yaml"
        # copy tạm để Ultralytics tìm được (nó sẽ tìm theo tên trong cwd)
        target = arch_file.with_name(model_arg)
        if not target.exists():
            target.write_text(arch_file.read_text())
    else:
        model_arg = str(arch_file)

    model = YOLO(model_arg)  # build architecture
    if args.pretrained:
        try:
            model.load(args.pretrained)  # transfer backbone weights nếu tương thích
            print(f"[improved] Loaded pretrained: {args.pretrained}")
        except Exception as e:
            print(f"[improved][WARN] Không load được pretrained '{args.pretrained}': {e}")

    if not args.no_class_weights:
        # Đọc nc và train path từ data yaml để không hard-code cho 1 dataset
        with open(args.data) as fh:
            dcfg = yaml.safe_load(fh)
        nc = int(dcfg.get("nc", 0)) or len(dcfg.get("names", []))
        base = Path(dcfg.get("path", args.data.parent))
        if not base.is_absolute():
            base = args.data.parent / base
        train_rel = dcfg.get("train", "images/train")
        labels_dir = base / str(train_rel).replace("images", "labels", 1)
        if labels_dir.exists() and nc > 0:
            cw = compute_class_weights(labels_dir, num_classes=nc)
            model.model.class_weights = cw
            print(f"[improved] class_weights (nc={nc}, dir={labels_dir}) = {cw.tolist()}")
        else:
            print(f"[improved][WARN] {labels_dir} không tồn tại hoặc nc=0 — bỏ class-weighted loss")

    if args.focal:
        print(f"[improved] Dùng FocalLoss(gamma={args.focal_gamma}, alpha={args.focal_alpha})")
        patch_focal_loss(args.focal_gamma, args.focal_alpha)

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache if args.cache else False,
        exist_ok=args.exist_ok,
        seed=42,
    )

    metrics = model.val(data=str(args.data), split="test",
                        project=args.project, name=f"{args.name}_test", plots=True)
    print("\n[improved] mAP50 =", metrics.box.map50, " mAP50-95 =", metrics.box.map)


if __name__ == "__main__":
    main()
