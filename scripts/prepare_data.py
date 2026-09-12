"""Chuẩn bị dữ liệu YOLO cho phát hiện phương tiện giao thông VN.

Hỗ trợ:
- Gộp nhiều dataset nguồn (Roboflow VN vehicle, VisDrone đã chuyển YOLO...) về
  cùng một tập nhãn 5 lớp: motorcycle, car, bus, truck, bicycle.
- Remap class-id theo mapping do người dùng khai báo trong configs/class_mapping.json
  (mỗi dataset một mapping riêng vì thứ tự lớp gốc khác nhau).
- Chia train/val/test theo tỉ lệ.
- In thống kê số object mỗi lớp (đo mức mất cân bằng).

Cấu trúc INPUT mong đợi (đặt vào data/raw/<source_name>/):
    data/raw/<source>/images/*.jpg
    data/raw/<source>/labels/*.txt   # định dạng YOLO: cls cx cy w h

Cấu trúc OUTPUT (data/processed):
    data/processed/images/{train,val,test}/*.jpg
    data/processed/labels/{train,val,test}/*.txt
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

# 5 lớp phương tiện VN (khớp data/data.yaml)
CANONICAL_CLASSES = ["motorcycle", "car", "bus", "truck", "bicycle"]
CLS_TO_ID = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

# Mapping mặc định gợi ý cho hai nguồn phổ biến. Người dùng có thể ghi đè
# bằng file JSON qua --mapping. Giá trị = None nghĩa là bỏ (không thuộc 5 lớp).
DEFAULT_MAPPING = {
    # Ví dụ Roboflow "Vietnamese vehicle" đã tách 5 lớp đúng:
    "roboflow_vn": {
        "0": "motorcycle",
        "1": "car",
        "2": "bus",
        "3": "truck",
        "4": "bicycle",
    },
    # VisDrone gốc có 10 lớp (pedestrian, people, bicycle, car, van, truck,
    # tricycle, awning-tricycle, bus, motor). Ta chỉ giữ các lớp phương tiện.
    "visdrone": {
        "0": None,             # pedestrian
        "1": None,             # people
        "2": "bicycle",
        "3": "car",
        "4": "car",            # van gộp vào car
        "5": "truck",
        "6": None,             # tricycle (không map)
        "7": None,             # awning-tricycle
        "8": "bus",
        "9": "motorcycle",
    },
}


def load_mapping(path: Path | None) -> dict:
    if path is None:
        return DEFAULT_MAPPING
    with open(path) as f:
        return json.load(f)


def remap_label_file(src: Path, dst: Path, mapping: dict[str, str | None]) -> int:
    """Đọc 1 file nhãn YOLO, remap class-id, ghi ra dst. Trả về số object giữ lại."""
    kept = 0
    lines_out = []
    with open(src) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_raw, *bbox = parts
            new_cls = mapping.get(cls_raw)
            if new_cls is None:
                continue
            new_id = CLS_TO_ID[new_cls]
            lines_out.append(f"{new_id} {' '.join(bbox)}\n")
            kept += 1
    if lines_out:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            f.writelines(lines_out)
    return kept


def collect_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """Ghép ảnh-nhãn theo tên file (bỏ pair nếu thiếu một trong hai)."""
    img_dir = source_dir / "images"
    lbl_dir = source_dir / "labels"
    pairs = []
    if not img_dir.exists():
        return pairs
    for img in sorted(img_dir.rglob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def split_pairs(pairs: list, ratios: tuple[float, float, float], seed: int):
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }


def count_classes(label_files: list[Path]) -> Counter:
    counter: Counter = Counter()
    for f in label_files:
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 5:
                    counter[int(parts[0])] += 1
    return counter


def print_stats(name: str, counter: Counter) -> None:
    total = sum(counter.values())
    print(f"\n=== {name}: {total} objects ===")
    if total == 0:
        return
    for cls_id, cls_name in enumerate(CANONICAL_CLASSES):
        n = counter.get(cls_id, 0)
        pct = 100 * n / total if total else 0.0
        bar = "#" * int(pct / 2)
        print(f"  [{cls_id}] {cls_name:12s} {n:8d}  ({pct:5.2f}%) {bar}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                   help="Thư mục chứa các dataset gốc (mỗi dataset một subfolder)")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--mapping", type=Path, default=None,
                   help="File JSON override class mapping cho từng source")
    p.add_argument("--sources", nargs="+", default=None,
                   help="Chỉ xử lý các source-name này (mặc định: tất cả subfolder)")
    p.add_argument("--split", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true",
                   help="Copy ảnh thay vì symlink (dùng khi chuyển volume/WSL mount)")
    p.add_argument("--stats-only", action="store_true",
                   help="Chỉ in thống kê của data/processed hiện có, không copy")
    args = p.parse_args()

    if args.stats_only:
        for split in ("train", "val", "test"):
            files = list((args.out_dir / "labels" / split).glob("*.txt"))
            print_stats(f"processed/{split}", count_classes(files))
        return

    mappings = load_mapping(args.mapping)
    if abs(sum(args.split) - 1.0) > 1e-6:
        raise SystemExit(f"Tổng split phải = 1.0, hiện {sum(args.split)}")

    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (args.out_dir / sub / split).mkdir(parents=True, exist_ok=True)

    all_pairs_by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    grand_counter_by_split: dict[str, Counter] = {k: Counter() for k in all_pairs_by_split}

    sources = args.sources or [p.name for p in args.raw_dir.iterdir() if p.is_dir()]
    if not sources:
        raise SystemExit(f"Không thấy source nào trong {args.raw_dir}")

    for src_name in sources:
        src_dir = args.raw_dir / src_name
        mapping = mappings.get(src_name)
        if mapping is None:
            print(f"[WARN] Bỏ qua '{src_name}' — chưa có mapping. "
                  f"Thêm vào configs/class_mapping.json hoặc --mapping.")
            continue
        pairs = collect_pairs(src_dir)
        if not pairs:
            print(f"[WARN] '{src_name}' không có cặp ảnh/nhãn.")
            continue
        splits = split_pairs(pairs, tuple(args.split), args.seed)
        print(f"[{src_name}] {len(pairs)} ảnh -> "
              f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

        for split, sp_pairs in splits.items():
            for img, lbl in sp_pairs:
                new_stem = f"{src_name}__{img.stem}"
                dst_img = args.out_dir / "images" / split / (new_stem + img.suffix.lower())
                dst_lbl = args.out_dir / "labels" / split / (new_stem + ".txt")
                kept = remap_label_file(lbl, dst_lbl, mapping)
                if kept == 0:
                    continue
                if dst_img.exists():
                    dst_img.unlink()
                if args.copy:
                    shutil.copy2(img, dst_img)
                else:
                    dst_img.symlink_to(img.resolve())
                all_pairs_by_split[split].append((dst_img, dst_lbl))

    for split in ("train", "val", "test"):
        files = [lbl for _, lbl in all_pairs_by_split[split]]
        grand_counter_by_split[split] = count_classes(files)
        print_stats(f"processed/{split}", grand_counter_by_split[split])

    # Xuất imbalance report ra results/tables/
    out_report = Path("results/tables/class_distribution.csv")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w") as f:
        f.write("split,class_id,class_name,count\n")
        for split, counter in grand_counter_by_split.items():
            for cls_id, cls_name in enumerate(CANONICAL_CLASSES):
                f.write(f"{split},{cls_id},{cls_name},{counter.get(cls_id, 0)}\n")
    print(f"\n[OK] Ghi thống kê -> {out_report}")


if __name__ == "__main__":
    main()
