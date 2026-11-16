#!/usr/bin/env bash
# Long-running, interruptible training supervisor.
#
# The cloud VM of this chat sleeps / gets recycled when a turn ends, so nothing
# here can run "forever" by itself. What this gives you instead is a loop that
# is *safe to kill at any moment*: every slice checkpoints, then the loop backs
# up to git and sleeps 5 minutes (so a restarted VM resumes where it stopped).
#
#   ./scripts/train_forever.sh                 # 25 min slices, sleeps 5 min
#   SLICE=20 TOTAL=200000 ./scripts/train_forever.sh
#
# Run it under nohup/tmux on a machine you control for a continuous run:
#   nohup ./scripts/train_forever.sh >> logs/supervisor.log 2>&1 &
set -u

SLICE="${SLICE:-25}"            # minutes of compute per slice
TOTAL="${TOTAL:-60000}"         # total iterations planned
BATCH="${BATCH:-8}"
CROP="${CROP:-192}"
THREADS="${THREADS:-8}"
DATA="${DATA:-/tmp/vfi/data/*.parquet}"
DERF="${DERF:-/tmp/vfi/derf}"
VAL_CLIPS="${VAL_CLIPS:-Johnny,shields}"
BACKUP_EVERY="${BACKUP_EVERY:-1800}"   # 30 min
PY="${PY:-/tmp/vfi/.venv/bin/python}"
LAST_BACKUP=$(date +%s)

mkdir -p logs checkpoints

echo "[supervisor] start $(date -Is) slice=${SLICE}min total=${TOTAL}it"
while :; do
  "$PY" -m src.train \
      --data $DATA --derf-dir "$DERF" --val-clips-names "$VAL_CLIPS" \
      --iters "$TOTAL" --max-minutes "$SLICE" \
      --batch "$BATCH" --crop "$CROP" --threads "$THREADS" --workers 0 \
      --resume checkpoints/ckpt_last.pt \
      --width "${WIDTH:-24,48,64}" \
      --val-every 400 --ckpt-every 200 2>&1 | tee -a logs/supervisor.log

  # keep the deployable artefacts fresh
  if [ -f checkpoints/ckpt_last.pt ]; then
    "$PY" -m src.export_onnx --ckpt checkpoints/ckpt_last.pt \
        --out checkpoints/mobileifnet.onnx >> logs/supervisor.log 2>&1 || true
  fi

  now=$(date +%s)
  if [ $((now - LAST_BACKUP)) -ge "$BACKUP_EVERY" ]; then
    ./scripts/backup.sh "auto: training checkpoint $(date -Is)" || true
    LAST_BACKUP=$now
  fi

  if [ -f checkpoints/ckpt_last.pt ]; then
    it=$("$PY" - <<'EOF'
import torch
try:
    print(torch.load("checkpoints/ckpt_last.pt", map_location="cpu", weights_only=False)["iter"])
except Exception:
    print(0)
EOF
)
    if [ "${it:-0}" -ge "$TOTAL" ]; then
      echo "[supervisor] reached $it/$TOTAL iterations, stopping $(date -Is)"
      break
    fi
  fi

  echo "[supervisor] slice done, sleeping 5 min $(date -Is)"
  sleep 300   # <- the "sleep every 5 minutes" tick
done
