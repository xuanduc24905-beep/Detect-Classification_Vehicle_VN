#!/usr/bin/env bash
# Wrapper chạy train YOLOv8 baseline / improved trên dataset DETRAC.
#
# Cách dùng:
#   scripts/train.sh improved            # train improved từ đầu (foreground)
#   scripts/train.sh improved bg         # train improved chạy nền, log vào logs/
#   scripts/train.sh resume              # resume improved từ last.pt
#   scripts/train.sh resume bg           # resume improved chạy nền
#   scripts/train.sh baseline            # train baseline
#   scripts/train.sh baseline bg         # train baseline chạy nền
#
# Đổi hyperparameters ở block CONFIG ngay dưới.

set -euo pipefail

# Chuyển vào project root (thư mục cha của scripts/)
cd "$(dirname "$0")/.."

# ========== CONFIG ==========
DATA=data/detrac.yaml
ARCH=models/yolov8_improved/yolov8n-cbam-detrac.yaml
PRETRAINED=weights/yolov8n.pt
SCALE=n
EPOCHS=50
IMGSZ=640
BATCH=16
WORKERS=8
CACHE=ram
DEVICE=0
PATIENCE=15
NAME=exp
PROJECT_BASE=runs
# ============================

MODE=${1:-improved}
BG=${2:-fg}

mkdir -p logs

run() {
    local cmd="$1"
    local logfile="$2"
    if [[ "$BG" == "bg" ]]; then
        echo "→ Chạy nền, log: $logfile"
        nohup bash -c "$cmd" > "$logfile" 2>&1 &
        echo "PID=$!  (tail -f $logfile để xem)"
    else
        eval "$cmd"
    fi
}

case "$MODE" in
    improved)
        LOG=logs/improved_$(date +%Y%m%d_%H%M%S).log
        CMD="python scripts/train_yolo_improved.py \
            --data $DATA \
            --arch $ARCH \
            --pretrained $PRETRAINED \
            --scale $SCALE \
            --epochs $EPOCHS \
            --imgsz $IMGSZ \
            --batch $BATCH \
            --workers $WORKERS \
            --cache $CACHE \
            --device $DEVICE \
            --patience $PATIENCE \
            --project $PROJECT_BASE/yolo_improved \
            --name $NAME \
            --exist-ok"
        run "$CMD" "$LOG"
        ;;

    resume)
        LOG=logs/improved_resume_$(date +%Y%m%d_%H%M%S).log
        LAST=$PROJECT_BASE/yolo_improved/$NAME/weights/last.pt
        if [[ ! -f "$LAST" ]]; then
            echo "ERROR: không thấy $LAST để resume" >&2
            exit 1
        fi
        CMD="python scripts/train_yolo_improved.py --resume \
            --project $PROJECT_BASE/yolo_improved --name $NAME"
        run "$CMD" "$LOG"
        ;;

    baseline)
        LOG=logs/baseline_$(date +%Y%m%d_%H%M%S).log
        CMD="python scripts/train_yolo_baseline.py \
            --weights $PRETRAINED \
            --data $DATA \
            --epochs $EPOCHS \
            --imgsz $IMGSZ \
            --batch $BATCH \
            --device $DEVICE \
            --patience $PATIENCE \
            --project $PROJECT_BASE/yolo_baseline \
            --name $NAME"
        run "$CMD" "$LOG"
        ;;

    *)
        echo "Unknown mode: $MODE" >&2
        echo "Dùng: $0 {improved|resume|baseline} [bg]" >&2
        exit 1
        ;;
esac
