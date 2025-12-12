#!/usr/bin/env python3
"""
stream_train.py — streaming FASTA fetch → encode → train pipeline for plasmids.

Subcommands:
  - init          : (re)initialize dirs + state
  - catalog-show  : show catalog contents
  - fetch-one     : fetch FASTA for a single accession
  - encode-one    : encode one accession into windows/tensors
  - train-one     : run a stub training loop on a single accession
  - stream        : full streaming loop over catalog using state

This version has a stub "trainer" that just pretends to train and logs info.
You can later swap in a real neural net (PyTorch, etc.) where marked.
"""

import argparse
import json
import os
import sys
import time
import random
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import requests

try:
    import yaml  # type: ignore
except ImportError:  # optional
    yaml = None


# ---------------------------------------------------------------------------
# Defaults & config helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "ncbi": {
        "email": "you@example.com",
        "api_key": None,
        "max_retries": 3,
        "backoff_seconds": 2.0,
    },
    "training": {
        "steps_per_plasmid": 50,
        "batch_size": 16,
        "window_size": 512,
        "stride": 256,
        "max_stream_epochs": 100,
        "shuffle_catalog": True,
    },
    "io": {
        "cache_fasta_dir": "cache/fasta",
        "cache_encoded_dir": "cache/encoded",
        "model_dir": "model",
        "checkpoints_dir": "model/checkpoints",
        "logs_dir": "logs",
        "state_file": "state/progress.json",
    },
}


@dataclass
class NCBIConfig:
    email: str
    api_key: Optional[str]
    max_retries: int
    backoff_seconds: float


@dataclass
class TrainingConfig:
    steps_per_plasmid: int
    batch_size: int
    window_size: int
    stride: int
    max_stream_epochs: int
    shuffle_catalog: bool


@dataclass
class IOConfig:
    cache_fasta_dir: str
    cache_encoded_dir: str
    model_dir: str
    checkpoints_dir: str
    logs_dir: str
    state_file: str


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge updates into base."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_full_config(path: str) -> Dict[str, Any]:
    """Load YAML config if available; fall back to defaults."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy via JSON
    if yaml is not None and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} did not contain a dict.")
        deep_update(cfg, data)
    elif not os.path.exists(path):
        print(f"[config] {path} not found; using built-in defaults.", file=sys.stderr)
    else:
        print(
            f"[config] PyYAML not installed; ignoring {path} and using defaults.",
            file=sys.stderr,
        )
    return cfg


def extract_configs(cfg: Dict[str, Any]) -> (NCBIConfig, TrainingConfig, IOConfig):
    n = cfg["ncbi"]
    t = cfg["training"]
    io = cfg["io"]
    return (
        NCBIConfig(
            email=n.get("email", "you@example.com"),
            api_key=n.get("api_key"),
            max_retries=int(n.get("max_retries", 3)),
            backoff_seconds=float(n.get("backoff_seconds", 2.0)),
        ),
        TrainingConfig(
            steps_per_plasmid=int(t.get("steps_per_plasmid", 50)),
            batch_size=int(t.get("batch_size", 16)),
            window_size=int(t.get("window_size", 512)),
            stride=int(t.get("stride", 256)),
            max_stream_epochs=int(t.get("max_stream_epochs", 100)),
            shuffle_catalog=bool(t.get("shuffle_catalog", True)),
        ),
        IOConfig(
            cache_fasta_dir=io.get("cache_fasta_dir", "cache/fasta"),
            cache_encoded_dir=io.get("cache_encoded_dir", "cache/encoded"),
            model_dir=io.get("model_dir", "model"),
            checkpoints_dir=io.get("checkpoints_dir", "model/checkpoints"),
            logs_dir=io.get("logs_dir", "logs"),
            state_file=io.get("state_file", "state/progress.json"),
        ),
    )


# ---------------------------------------------------------------------------
# Catalog, state, logging
# ---------------------------------------------------------------------------

def read_catalog(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalog file not found: {path}")
    accessions: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # first token is accession
            acc = line.split()[0]
            accessions.append(acc)
    if not accessions:
        raise ValueError(f"Catalog {path} contained no accessions.")
    return accessions


def ensure_dirs(io_cfg: IOConfig) -> None:
    os.makedirs(io_cfg.cache_fasta_dir, exist_ok=True)
    os.makedirs(io_cfg.cache_encoded_dir, exist_ok=True)
    os.makedirs(io_cfg.model_dir, exist_ok=True)
    os.makedirs(io_cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(io_cfg.logs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(io_cfg.state_file), exist_ok=True)


def load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "current_index": 0,
            "total_steps": 0,
            "plasmid_visit_counts": {},
            "last_checkpoint": None,
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str, state: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def setup_logging(logs_dir: str) -> None:
    os.makedirs(logs_dir, exist_ok=True)
    train_log = os.path.join(logs_dir, "training.log")
    fetch_log = os.path.join(logs_dir, "fetch.log")

    # Basic root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(train_log, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Separate fetch logger
    fetch_handler = logging.FileHandler(fetch_log, mode="a")
    fetch_logger = logging.getLogger("fetch")
    fetch_logger.setLevel(logging.INFO)
    fetch_logger.addHandler(fetch_handler)


# ---------------------------------------------------------------------------
# NCBI fetching
# ---------------------------------------------------------------------------

def fetch_fasta(
    accession: str,
    io_cfg: IOConfig,
    ncbi_cfg: NCBIConfig,
    force: bool = False,
) -> str:
    """Download FASTA for accession from NCBI nuccore, returning local path."""
    out_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
    fetch_logger = logging.getLogger("fetch")

    if os.path.exists(out_path) and not force:
        fetch_logger.info(f"{accession}: using cached FASTA at {out_path}")
        return out_path

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "nuccore",
        "id": accession,
        "rettype": "fasta",
        "retmode": "text",
        "email": ncbi_cfg.email,
    }
    if ncbi_cfg.api_key:
        params["api_key"] = ncbi_cfg.api_key

    fetch_logger.info(f"{accession}: fetching from NCBI {url}")
    last_error = None
    for attempt in range(ncbi_cfg.max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200 and resp.text.strip().startswith(">"):
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                fetch_logger.info(
                    f"{accession}: fetched and saved FASTA ({len(resp.text)} bytes)"
                )
                return out_path
            else:
                msg = (
                    f"HTTP {resp.status_code}, "
                    f"first 80 chars: {resp.text[:80]!r}"
                )
                last_error = msg
        except Exception as e:  # noqa: BLE001
            last_error = str(e)

        if attempt < ncbi_cfg.max_retries:
            delay = ncbi_cfg.backoff_seconds * (2 ** attempt)
            fetch_logger.warning(
                f"{accession}: fetch attempt {attempt+1} failed ({last_error}); "
                f"retrying in {delay:.1f}s"
            )
            time.sleep(delay)

    raise RuntimeError(f"Failed to fetch {accession} from NCBI: {last_error}")


# ---------------------------------------------------------------------------
# FASTA parsing & encoding
# ---------------------------------------------------------------------------

def parse_fasta_sequence(path: str) -> str:
    """Very simple FASTA parser: returns concatenated sequence (first record)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"FASTA not found: {path}")
    seq_lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
    seq = "".join(seq_lines).upper()
    if not seq:
        raise ValueError(f"No sequence found in {path}")
    return seq


def encode_sequence_one_hot(
    seq: str,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Encode sequence as one-hot windows: shape (num_windows, window_size, 4)."""
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    length = len(seq)

    if length < window_size:
        # Single padded window
        arr = np.zeros((window_size, 4), dtype=np.float32)
        for i, base in enumerate(seq):
            idx = base_to_idx.get(base)
            if idx is not None:
                arr[i, idx] = 1.0
        return arr[None, ...]  # shape (1, window_size, 4)

    windows: List[np.ndarray] = []
    for start in range(0, length - window_size + 1, stride):
        window_seq = seq[start : start + window_size]
        arr = np.zeros((window_size, 4), dtype=np.float32)
        for i, base in enumerate(window_seq):
            idx = base_to_idx.get(base)
            if idx is not None:
                arr[i, idx] = 1.0
        windows.append(arr)

    if not windows:
        # Fallback to single padded window if stride was weird
        arr = np.zeros((window_size, 4), dtype=np.float32)
        for i, base in enumerate(seq[:window_size]):
            idx = base_to_idx.get(base)
            if idx is not None:
                arr[i, idx] = 1.0
        windows.append(arr)

    return np.stack(windows, axis=0)


def encode_accession(
    accession: str,
    io_cfg: IOConfig,
    window_size: int,
    stride: int,
    save_to_disk: bool = True,
) -> np.ndarray:
    """Load FASTA, encode to one-hot windows. Optionally save .npy file."""
    fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA for {accession} not found at {fasta_path}")

    seq = parse_fasta_sequence(fasta_path)
    encoded = encode_sequence_one_hot(seq, window_size, stride)
    logging.info(
        f"{accession}: encoded sequence of length {len(seq)} -> "
        f"{encoded.shape[0]} windows, window_size={window_size}, stride={stride}"
    )

    if save_to_disk:
        os.makedirs(io_cfg.cache_encoded_dir, exist_ok=True)
        out_path = os.path.join(io_cfg.cache_encoded_dir, f"{accession}.npy")
        np.save(out_path, encoded)
        logging.info(f"{accession}: saved encoded tensor to {out_path}")

    return encoded


# ---------------------------------------------------------------------------
# Stub training logic
# ---------------------------------------------------------------------------

def stub_train_on_encoded(
    accession: str,
    encoded: np.ndarray,
    steps: int,
    batch_size: int,
    state: Dict[str, Any],
) -> float:
    """
    Stub trainer: simulate `steps` of training and update state["total_steps"].
    Returns a fake "loss" for logging.

    Replace this with real ML (PyTorch, etc.) later.
    """
    num_windows = encoded.shape[0]
    total_steps = state.get("total_steps", 0)

    logging.info(
        f"{accession}: starting stub training on tensor "
        f"shape={encoded.shape}, steps={steps}, batch_size={batch_size}"
    )

    # Silly fake loss that shrinks with total steps
    fake_loss = 1.0 / (1.0 + total_steps + steps)

    # Pretend to iterate (we don't want this to be slow)
    state["total_steps"] = total_steps + steps
    logging.info(
        f"{accession}: finished stub training; "
        f"total_steps={state['total_steps']}, fake_loss={fake_loss:.6f}"
    )
    return fake_loss


def cleanup_accession_files(accession: str, io_cfg: IOConfig) -> None:
    """Delete FASTA and encoded files for accession (if they exist)."""
    fasta_path = os.path.join(io_cfg.cache_fasta_dir, f"{accession}.fasta")
    encoded_path = os.path.join(io_cfg.cache_encoded_dir, f"{accession}.npy")
    for path in (fasta_path, encoded_path):
        if os.path.exists(path):
            try:
                os.remove(path)
                logging.info(f"{accession}: deleted {path}")
            except OSError as e:
                logging.warning(f"{accession}: failed to delete {path}: {e}")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, _, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)

    # Initialize state if missing
    state = load_state(io_cfg.state_file)
    save_state(io_cfg.state_file, state)
    print(f"Initialized project. State file at: {io_cfg.state_file}")
    return 0


def cmd_catalog_show(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, _, _ = extract_configs(cfg)
    accessions = read_catalog(args.catalog)

    print(f"Catalog: {args.catalog}")
    for idx, acc in enumerate(accessions):
        print(f"[{idx}] {acc}")
    print(f"Total accessions: {len(accessions)}")
    return 0


def cmd_fetch_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    ncbi_cfg, _, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    path = fetch_fasta(args.accession, io_cfg, ncbi_cfg, force=args.force)
    print(f"Fetched {args.accession} -> {path}")
    return 0


def cmd_encode_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    _, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    window_size = args.window_size or train_cfg.window_size
    stride = args.stride or train_cfg.stride

    encoded = encode_accession(
        args.accession,
        io_cfg,
        window_size=window_size,
        stride=stride,
        save_to_disk=not args.no_save,
    )
    print(
        f"{args.accession}: encoded tensor shape {encoded.shape}, "
        f"window_size={window_size}, stride={stride}"
    )
    return 0


def cmd_train_one(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    window_size = args.window_size or train_cfg.window_size
    stride = args.stride or train_cfg.stride
    steps = args.steps or train_cfg.steps_per_plasmid
    batch_size = args.batch_size or train_cfg.batch_size

    # Load or init state
    state = load_state(io_cfg.state_file)

    # Fetch + encode
    fetch_fasta(args.accession, io_cfg, ncbi_cfg, force=args.force)
    encoded = encode_accession(
        args.accession,
        io_cfg,
        window_size=window_size,
        stride=stride,
        save_to_disk=not args.no_save,
    )

    # Stub train
    stub_train_on_encoded(args.accession, encoded, steps, batch_size, state)

    # Update visit counts
    pvc = state.setdefault("plasmid_visit_counts", {})
    pvc[args.accession] = pvc.get(args.accession, 0) + 1
    save_state(io_cfg.state_file, state)

    # Cleanup if requested
    if args.cleanup:
        cleanup_accession_files(args.accession, io_cfg)

    print(
        f"Completed stub train-one for {args.accession}; "
        f"total_steps={state['total_steps']}"
    )
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    cfg = load_full_config(args.config)
    ncbi_cfg, train_cfg, io_cfg = extract_configs(cfg)
    ensure_dirs(io_cfg)
    setup_logging(io_cfg.logs_dir)

    accessions = read_catalog(args.catalog)
    state = load_state(io_cfg.state_file)

    # CLI overrides
    steps_per_plasmid = args.steps_per_plasmid or train_cfg.steps_per_plasmid
    batch_size = args.batch_size or train_cfg.batch_size
    window_size = args.window_size or train_cfg.window_size
    stride = args.stride or train_cfg.stride
    max_epochs = args.max_stream_epochs or train_cfg.max_stream_epochs
    shuffle_catalog = (
        train_cfg.shuffle_catalog if args.shuffle is None else args.shuffle
    )

    current_index = state.get("current_index", 0)
    num_acc = len(accessions)

    logging.info(
        f"Starting streaming loop: {num_acc} accessions, "
        f"max_epochs={max_epochs}, steps_per_plasmid={steps_per_plasmid}, "
        f"window_size={window_size}, stride={stride}, "
        f"shuffle_catalog={shuffle_catalog}"
    )

    for epoch in range(max_epochs):
        indices = list(range(num_acc))
        if shuffle_catalog:
            random.shuffle(indices)
        else:
            # Start from current_index on first epoch
            if epoch == 0 and 0 <= current_index < num_acc:
                indices = list(range(current_index, num_acc)) + list(
                    range(0, current_index)
                )

        logging.info(f"Epoch {epoch+1}/{max_epochs}: order={indices}")

        for idx in indices:
            accession = accessions[idx]
            logging.info(f"Streaming accession idx={idx}, id={accession}")

            # Fetch + encode
            fetch_fasta(accession, io_cfg, ncbi_cfg, force=args.force)
            encoded = encode_accession(
                accession,
                io_cfg,
                window_size=window_size,
                stride=stride,
                save_to_disk=not args.no_save,
            )

            # Train (stub)
            stub_train_on_encoded(accession, encoded, steps_per_plasmid, batch_size, state)

            # Update state
            pvc = state.setdefault("plasmid_visit_counts", {})
            pvc[accession] = pvc.get(accession, 0) + 1
            state["current_index"] = idx
            save_state(io_cfg.state_file, state)

            # Cleanup
            if args.cleanup:
                cleanup_accession_files(accession, io_cfg)

        # You can add early stopping / conditions here later

    logging.info("Streaming loop finished.")
    return 0


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="genostream streaming FASTA trainer (stub version)",
    )
    parser.add_argument(
        "--config",
        default="config/stream_config.yaml",
        help="Path to YAML config file (default: config/stream_config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Initialize directories and state.")
    p_init.set_defaults(func=cmd_init)

    # catalog-show
    p_cat = subparsers.add_parser(
        "catalog-show", help="Show accessions in the catalog."
    )
    p_cat.add_argument(
        "--catalog",
        default="config/plasmids_5.txt",
        help="Catalog file containing accessions (default: config/plasmids_5.txt)",
    )
    p_cat.set_defaults(func=cmd_catalog_show)

    # fetch-one
    p_fetch = subparsers.add_parser(
        "fetch-one", help="Fetch FASTA for a single accession."
    )
    p_fetch.add_argument("accession", help="Accession ID to fetch.")
    p_fetch.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if FASTA exists.",
    )
    p_fetch.set_defaults(func=cmd_fetch_one)

    # encode-one
    p_enc = subparsers.add_parser(
        "encode-one", help="Encode one accession into one-hot windows."
    )
    p_enc.add_argument("accession", help="Accession ID to encode.")
    p_enc.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override window size (default from config).",
    )
    p_enc.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override stride (default from config).",
    )
    p_enc.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save encoded tensor to disk.",
    )
    p_enc.set_defaults(func=cmd_encode_one)

    # train-one
    p_train = subparsers.add_parser(
        "train-one", help="Fetch + encode + stub-train on a single accession."
    )
    p_train.add_argument("accession", help="Accession ID to train on.")
    p_train.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Steps for this accession (default from config).",
    )
    p_train.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default from config).",
    )
    p_train.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override window size (default from config).",
    )
    p_train.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override stride (default from config).",
    )
    p_train.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of FASTA.",
    )
    p_train.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save encoded tensor to disk.",
    )
    p_train.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete FASTA/encoded files after training.",
    )
    p_train.set_defaults(func=cmd_train_one)

    # stream
    p_stream = subparsers.add_parser(
        "stream", help="Full streaming loop over the catalog."
    )
    p_stream.add_argument(
        "--catalog",
        default="config/plasmids_5.txt",
        help="Catalog file containing accessions (default: config/plasmids_5.txt)",
    )
    p_stream.add_argument(
        "--steps-per-plasmid",
        type=int,
        default=None,
        help="Steps per plasmid visit (default from config).",
    )
    p_stream.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default from config).",
    )
    p_stream.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override window size (default from config).",
    )
    p_stream.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override stride (default from config).",
    )
    p_stream.add_argument(
        "--max-stream-epochs",
        type=int,
        default=None,
        help="Number of passes over catalog (default from config).",
    )
    p_stream.add_argument(
        "--shuffle",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Shuffle catalog each epoch (true/false, default from config).",
    )
    p_stream.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of FASTAs.",
    )
    p_stream.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save encoded tensors to disk.",
    )
    p_stream.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete FASTA/encoded files after each plasmid.",
    )
    p_stream.set_defaults(func=cmd_stream)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

