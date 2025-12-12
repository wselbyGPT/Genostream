#!/usr/bin/env bash
# build.sh — bootstrap the genostream project tree
# Run from: /home/wselby/genostream

set -euo pipefail

# Root directory (assume we're already in /home/wselby/genostream)
ROOT_DIR="$(pwd)"

echo "Initializing genostream project tree in: $ROOT_DIR"

# ---------------------------------------------------------------------
# Create directory structure
# ---------------------------------------------------------------------
mkdir -p "$ROOT_DIR/config"
mkdir -p "$ROOT_DIR/state"
mkdir -p "$ROOT_DIR/cache/fasta"
mkdir -p "$ROOT_DIR/cache/encoded"
mkdir -p "$ROOT_DIR/model/checkpoints"
mkdir -p "$ROOT_DIR/logs"

# ---------------------------------------------------------------------
# Create config/plasmids_5.txt (only if it doesn't exist)
# ---------------------------------------------------------------------
PLASMID_LIST="$ROOT_DIR/config/plasmids_5.txt"
if [[ ! -f "$PLASMID_LIST" ]]; then
  cat > "$PLASMID_LIST" << 'EOF'
# Five example plasmid accessions for streaming prototype
# accession    # comment (human-readable name)
L09137.2      # pUC19
J01749.1      # pBR322
U13853.1      # pGEX-4T-1
EF442785.1    # pET28a-LIC
CP060383.1    # pRK100 (large conjugative plasmid)
EOF
  echo "Created: $PLASMID_LIST"
else
  echo "Exists, not overwriting: $PLASMID_LIST"
fi

# ---------------------------------------------------------------------
# Create config/stream_config.yaml (only if it doesn't exist)
# ---------------------------------------------------------------------
STREAM_CONFIG="$ROOT_DIR/config/stream_config.yaml"
if [[ ! -f "$STREAM_CONFIG" ]]; then
  cat > "$STREAM_CONFIG" << 'EOF'
# stream_config.yaml — default configuration for genostream

ncbi:
  email: "you@example.com"      # TODO: set to your email for NCBI
  api_key: null                 # optional NCBI API key
  max_retries: 3
  backoff_seconds: 2

training:
  steps_per_plasmid: 50         # gradient steps per plasmid visit
  batch_size: 16
  window_size: 512
  stride: 256
  max_stream_epochs: 100
  shuffle_catalog: true

io:
  cache_fasta_dir: "cache/fasta"
  cache_encoded_dir: "cache/encoded"
  model_dir: "model"
  checkpoints_dir: "model/checkpoints"
  logs_dir: "logs"
  state_file: "state/progress.json"
EOF
  echo "Created: $STREAM_CONFIG"
else
  echo "Exists, not overwriting: $STREAM_CONFIG"
fi

# ---------------------------------------------------------------------
# Initialize state/progress.json (only if it doesn't exist)
# ---------------------------------------------------------------------
STATE_FILE="$ROOT_DIR/state/progress.json"
if [[ ! -f "$STATE_FILE" ]]; then
  cat > "$STATE_FILE" << 'EOF'
{
  "current_index": 0,
  "total_steps": 0,
  "plasmid_visit_counts": {},
  "last_checkpoint": null
}
EOF
  echo "Created: $STATE_FILE"
else
  echo "Exists, not overwriting: $STATE_FILE"
fi

# ---------------------------------------------------------------------
# Touch log files
# ---------------------------------------------------------------------
TRAIN_LOG="$ROOT_DIR/logs/training.log"
FETCH_LOG="$ROOT_DIR/logs/fetch.log"

touch "$TRAIN_LOG"
touch "$FETCH_LOG"

echo "Ensured log files:"
echo "  $TRAIN_LOG"
echo "  $FETCH_LOG"

# ---------------------------------------------------------------------
# Create a stub stream_train.py (only if it doesn't exist)
# ---------------------------------------------------------------------
STREAM_TRAIN="$ROOT_DIR/stream_train.py"
if [[ ! -f "$STREAM_TRAIN" ]]; then
  cat > "$STREAM_TRAIN" << 'EOF'
#!/usr/bin/env python3
"""
stream_train.py — CLI entrypoint for the genostream streaming trainer.

Subcommands to implement (design only for now):
  - init
  - catalog-show
  - fetch-one
  - encode-one
  - train-one
  - stream
"""

import sys


def main() -> int:
    # TODO: implement argparse + subcommands based on the architecture design.
    print("stream_train.py stub — implement CLI here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF
  chmod +x "$STREAM_TRAIN"
  echo "Created stub: $STREAM_TRAIN"
else
  echo "Exists, not overwriting: $STREAM_TRAIN"
fi

echo "genostream project tree initialized."
