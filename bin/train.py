#!/usr/bin/env python3
"""Orchestrate BasicModel training: embeddings → model.

Usage:
    python train.py                           # local, defaults
    python train.py --model data/BasicModel.xml --max-docs 500 --random-shards
    python train.py --host arbormini.local    # remote execution
"""

import argparse
import cProfile
import datetime
import os
import pstats
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util import TheMessage


def parse_args():
    p = argparse.ArgumentParser(description="Train BasicModel end-to-end")
    p.add_argument("--model", "-m", default="data/BasicModel.xml",
                   help="XML config file (default: data/BasicModel.xml)")
    p.add_argument("--data", default=None,
                   help="Dataset name (e.g. text, mnist, xor). "
                        "Overrides <dataset> in XML config.")
    p.add_argument("--max-docs", type=int, default=None,
                   help="Override maxDocs from XML config")
    p.add_argument("--num-shards", type=int, default=None,
                   help="Override numShards from XML config")
    p.add_argument("--num-epochs", type=int, default=None,
                   help="Override numEpochs from XML config")
    p.add_argument("--random-shards", action="store_true",
                   help="Pick random shard indices for variety across runs")
    p.add_argument("--force-embeddings", action="store_true",
                   help="Retrain embeddings even if they already exist")
    p.add_argument("--log", nargs="?", const="auto", default=None,
                   metavar="FILE",
                   help="Capture stdout/stderr to a .log file. "
                        "Omit filename for auto-generated timestamped name.")
    p.add_argument("--profile", action="store_true",
                   help="Run Phase 2 under cProfile; writes a .prof file "
                        "and prints the top-30 cumulative-time functions.")

    # SSH remote execution
    ssh = p.add_argument_group("remote execution (SSH)")
    ssh.add_argument("--host", default=None,
                     help="Remote host (e.g. arbormini.local)")
    ssh.add_argument("--user", default="arogers",
                     help="SSH user (default: arogers)")
    ssh.add_argument("--key-file", default="~/.ssh/id_ed25519_arbormini",
                     help="SSH key file")
    ssh.add_argument("--remote-dir", default="~/WikiOracle/basicmodel",
                     help="Remote working directory")
    return p.parse_args()


def project_dir():
    """Return basicmodel project root."""
    return str(Path(__file__).resolve().parent.parent)


_log_file = None   # set by main() when --log is active


def run(cmd, **kwargs):
    """Run a command, printing it first. Tees output to log file if active."""
    TheMessage(f"+ {' '.join(cmd)}")

    if _log_file:
        # Subprocess stdout goes to a pipe, not our Python-level _Tee,
        # so we stream it line-by-line through sys.stdout (the _Tee).
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, **kwargs)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        proc.wait()
        if proc.returncode != 0:
            sys.exit(proc.returncode)
    else:
        result = subprocess.run(cmd, **kwargs)
        if result.returncode != 0:
            sys.exit(result.returncode)


def read_xml_config(xml_path):
    """Read embedding and training params from the XML config."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    arch = root.find("architecture")
    result = {}
    if arch is not None:
        ep = arch.findtext("embeddingPath")
        if ep:
            result["embeddingPath"] = ep
    return result


def train_local(args):
    """Run training locally: embeddings then model."""
    proj = project_dir()
    python = os.path.join(proj, ".venv", "bin", "python")
    env = {**os.environ, "PYTHONPATH": os.path.join(proj, "bin"),
           "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
           "PYTHONUNBUFFERED": "1"}

    # Resolve the XML config path
    xml_path = args.model
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(proj, xml_path)

    # Read embeddingPath from XML — output embeddings where BasicModel.py expects them
    cfg = read_xml_config(xml_path)
    emb_relpath = cfg.get("embeddingPath")
    if emb_relpath:
        emb_path = os.path.join(os.path.dirname(xml_path), emb_relpath)
    else:
        emb_path = os.path.join(proj, "output", "embeddings", "sentence.pt")

    # --- Phase 1: Embeddings ---
    if args.force_embeddings or not os.path.exists(emb_path):
        embed_cmd = [
            python, os.path.join(proj, "bin", "embed.py"), "train",
            "--config", os.path.join(proj, "data", "sentence.cfg"),
            "--output", emb_path,
        ]
        if args.num_shards is not None:
            embed_cmd += ["--num-shards", str(args.num_shards)]
        if args.max_docs is not None:
            embed_cmd += ["--max-docs", str(args.max_docs)]
        if args.random_shards:
            embed_cmd += ["--random-shards"]
        TheMessage(f"\n=== Phase 1: Training embeddings → {emb_path} ===")
        run(embed_cmd, cwd=proj, env=env)
    else:
        TheMessage(f"\n=== Phase 1: Embeddings exist at {emb_path}, skipping ===")

    # --- Phase 2: Model training ---
    # Pass overrides via env so BasicModel.py respects them over XML config
    model_env = dict(env)
    if args.data is not None:
        model_env["BASIC_DATASET"] = args.data
    if args.max_docs is not None:
        model_env["BASIC_MAX_DOCS"] = str(args.max_docs)
    if args.num_shards is not None:
        model_env["BASIC_NUM_SHARDS"] = str(args.num_shards)
    if args.num_epochs is not None:
        model_env["BASIC_NUM_EPOCHS"] = str(args.num_epochs)

    if args.profile:
        # Run BasicModel.py under cProfile via -m cProfile -o <file>
        prof_dir = os.path.join(proj, "output", "profiles")
        os.makedirs(prof_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prof_path = os.path.join(prof_dir, f"train_{ts}.prof")
        model_cmd = [
            python, "-m", "cProfile", "-o", prof_path,
            os.path.join(proj, "bin", "BasicModel.py"), args.model,
        ]
        TheMessage(f"\n=== Phase 2: Training model (profiling → {prof_path}) ===")
        run(model_cmd, cwd=os.path.join(proj, "bin"), env=model_env)

        # Print summary
        TheMessage(f"\n=== Profile summary (top 30 by cumulative time) ===")
        stats = pstats.Stats(prof_path, stream=sys.stdout)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        TheMessage(f"\nFull profile saved to {prof_path}")
        TheMessage("View interactively:  snakeviz " + prof_path)
    else:
        model_cmd = [python, os.path.join(proj, "bin", "BasicModel.py"), args.model]
        TheMessage("\n=== Phase 2: Training model ===")
        run(model_cmd, cwd=os.path.join(proj, "bin"), env=model_env)


def train_remote(args):
    """Rsync project to remote host and run training there."""
    proj = project_dir()
    key = os.path.expanduser(args.key_file)
    ssh_opts = f"ssh -i {key}"
    dest = f"{args.user}@{args.host}:{args.remote_dir}/"

    # Rsync
    TheMessage(f"\n=== Syncing to {args.host} ===")
    rsync_cmd = [
        "rsync", "-av", "--progress",
        "-e", ssh_opts,
        "--exclude", ".venv",
        "--exclude", "__pycache__",
        "--exclude", ".DS_Store",
        "--exclude", "output/",
        f"{proj}/", dest,
    ]
    run(rsync_cmd)

    # Build remote command
    remote_args = ["bin/train.py", "--model", args.model]
    if args.data is not None:
        remote_args += ["--data", args.data]
    if args.max_docs is not None:
        remote_args += ["--max-docs", str(args.max_docs)]
    if args.num_shards is not None:
        remote_args += ["--num-shards", str(args.num_shards)]
    if args.num_epochs is not None:
        remote_args += ["--num-epochs", str(args.num_epochs)]
    if args.random_shards:
        remote_args += ["--random-shards"]
    if args.profile:
        remote_args += ["--profile"]
    # SSH and run — forward selected env vars that affect training behaviour
    remote_env_vars = "PYTHONUNBUFFERED=1 PYTHONPATH=bin"
    for var in ("BASICMODEL_DEVICE", "BASICMODEL_COMPILE"):
        val = os.environ.get(var)
        if val:
            remote_env_vars = f"{var}={val} {remote_env_vars}"

    TheMessage(f"\n=== Running training on {args.host} ===")
    remote_cmd = f"cd {args.remote_dir} && {remote_env_vars} .venv/bin/python {' '.join(remote_args)}"
    ssh_cmd = [
        "ssh", "-i", key,
        f"{args.user}@{args.host}",
        remote_cmd,
    ]
    run(ssh_cmd)


class _Tee:
    """Write to both an original stream and a log file."""
    def __init__(self, original, log):
        self._original = original
        self._log = log

    def write(self, data):
        self._original.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._original.flush()
        self._log.flush()

    def fileno(self):
        return self._original.fileno()


def main():
    global _log_file
    args = parse_args()

    if args.log is not None:
        proj = project_dir()
        log_dir = os.path.join(proj, "output", "logs")
        os.makedirs(log_dir, exist_ok=True)

        if args.log == "auto":
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(log_dir, f"train_{ts}.log")
        else:
            log_path = args.log if os.path.isabs(args.log) else os.path.join(log_dir, args.log)

        _log_file = open(log_path, "w")
        sys.stdout = _Tee(sys.__stdout__, _log_file)
        sys.stderr = _Tee(sys.__stderr__, _log_file)
        TheMessage(f"Logging to {log_path}")

    try:
        if args.host:
            train_remote(args)
        else:
            train_local(args)
    finally:
        if _log_file:
            _log_file.close()


if __name__ == "__main__":
    main()
