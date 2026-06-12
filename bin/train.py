#!/usr/bin/env python3
"""Orchestrate BasicModel training: embeddings -> model.

Usage:
    python train.py                           # local, defaults
    python train.py --model data/MM_20M.xml --compile-target gpu --batches 10
    python train.py --model data/MM_20M.xml --compile-target mlx
    python train.py --host example.org        # remote execution via SSH
"""

import argparse
import cProfile
import datetime
import os
import pstats
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util import TheMessage

WEIGHT_PATTERNS = ("*.ckpt", "*.kv")


def parse_args(argv=None):
    """Build the argparse spec for the train.py CLI.

    Covers Phase 1 / Phase 2 overrides (data, batch / token / epoch
    caps), torch.compile mode, profiling toggle, and the SSH remote-
    execution argument group. Returns the parsed Namespace.
    """
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
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Cap AR training/eval positions per document for "
                        "quick smoke runs. Full training uses the XML "
                        "InputSpace.nOutput cap when omitted.")
    p.add_argument("--batches", type=int, default=None, metavar="N",
                   help="Cap the training pass at N batches and stop "
                        "(across all epochs combined). Useful for "
                        "wall-clock-bounded runs: at ~7s/batch on GB10, "
                        "12000 batches is roughly 24 hours.")
    p.add_argument("--test", nargs="?", const=-1, type=int, default=None,
                   metavar="N",
                   help="Run the baseline + post-train test/validation "
                        "passes. Without this flag those passes are "
                        "skipped (training only) -- avoids the long "
                        "test-split traversal that can dominate wall-"
                        "clock at large maxDocs. Pass an integer "
                        "(e.g. --test 100) to cap each test pass at N "
                        "batches; --test alone runs the full passes.")
    p.add_argument("--compile-mode", default=None,
                   choices=("default", "reduce-overhead", "max-autotune",
                            "max-autotune-no-cudagraphs"),
                   help="torch.compile mode (sets MODEL_COMPILE_MODE env). "
                        "default = Inductor kernel fusion only (no "
                        "CUDAGraphs); reduce-overhead = Inductor + "
                        "CUDAGraphs; max-autotune = Inductor + "
                        "CUDAGraphs + autotune; max-autotune-no-"
                        "cudagraphs = autotune without CUDAGraphs. "
                        "Omitted means use MODEL_COMPILE_MODE or util.py's "
                        "runtime default.")
    p.add_argument("--compile-target", default="gpu",
                   choices=("gpu", "mlx"),
                   help="Compilation target. gpu runs normal training with "
                        "BASICMODEL_DEVICE=gpu and MODEL_COMPILE=auto by "
                        "default. mlx exports the model tensor core to an "
                        "ExecuTorch/MLX .pte and skips Phase 2 training.")
    p.add_argument("--mlx-output", default=None,
                   help="Output .pte path for --compile-target mlx. "
                        "Defaults to output/mlx/<model-stem>.pte.")
    p.add_argument("--random-shards", action="store_true",
                   help="Pick random shard indices for variety across runs")
    p.add_argument("--force-embeddings", action="store_true",
                   help="Retrain embeddings even if they already exist")
    p.add_argument("--latent-vector-size", type=int, default=None,
                   help="Phase 1 SBOW training dim. Forwarded to "
                        "embed.py train. Defaults (in embed.py) to "
                        "max(64, PerceptualSpace.nDim) -- SBOW trains "
                        "at the higher latent dim and PCA projects the "
                        "codebook to nDim for the saved artifact, giving "
                        "the codebook geometric room during training "
                        "while keeping the model at its configured nDim.")
    p.add_argument("--embed-lr", type=float, default=None,
                   help="Phase 1 SBOW learning rate, forwarded to "
                        "embed.py train. Default 0.01 (set by embed.py). "
                        "For manual annealing across runs, decrease this "
                        "between successive --force-embeddings invocations.")
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
                     help="Remote host")
    ssh.add_argument("--user", default="arogers",
                     help="SSH user (default: arogers)")
    ssh.add_argument("--key-file", default=None,
                     help="Optional SSH key file; uses SSH config/agent when omitted")
    ssh.add_argument("--remote-dir", default="~/WikiOracle/basicmodel",
                     help="Remote working directory")
    return p.parse_args(argv)


def project_dir():
    """Return basicmodel project root (one level above this file)."""
    return str(Path(__file__).resolve().parent.parent)


def venv_python(proj):
    """Return the in-project venv Python executable for this platform.

    ``BASICMODEL_PYTHON`` may point at an existing compatible environment,
    useful for isolated clones that intentionally do not duplicate a venv.
    Probes Windows ``Scripts/python.exe`` then POSIX ``bin/python``.
    Falls back to the platform-appropriate guess if neither file exists,
    so callers can still surface a useful error from subprocess.
    """
    override = os.environ.get("BASICMODEL_PYTHON")
    if override:
        return os.path.expanduser(override)

    candidates = [
        os.path.join(proj, ".venv", "Scripts", "python.exe"),
        os.path.join(proj, ".venv", "bin", "python"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if os.name == "nt" else candidates[1]


def resolve_model_path(proj, model_path):
    """Resolve a model XML path relative to the project root."""
    if os.path.isabs(model_path):
        return model_path
    return os.path.join(proj, model_path)


def default_mlx_output_path(proj, xml_path):
    """Return output/mlx/<model-stem>.pte for a resolved XML path."""
    stem = Path(xml_path).stem
    return os.path.join(proj, "output", "mlx", f"{stem}.pte")


def apply_compile_target_env(args, env):
    """Populate compile-target defaults without clobbering caller overrides."""
    if args.compile_target == "gpu":
        env.setdefault("BASICMODEL_DEVICE", "gpu")
        env.setdefault("MODEL_COMPILE", "auto")
    if args.compile_mode is not None:
        env["MODEL_COMPILE_MODE"] = args.compile_mode


def export_mlx_local(args, proj, python, xml_path):
    """Run the MLX export target and return the output .pte path."""
    output_pte = args.mlx_output
    if output_pte is None:
        output_pte = default_mlx_output_path(proj, xml_path)
    elif not os.path.isabs(output_pte):
        output_pte = os.path.join(proj, output_pte)

    os.makedirs(os.path.dirname(output_pte) or ".", exist_ok=True)
    export_env = {**os.environ, "PYTHONPATH": os.path.join(proj, "bin"),
                  "PYTHONUNBUFFERED": "1",
                  "BASICMODEL_DEVICE": "cpu",
                  "MODEL_COMPILE": "eager"}
    TheMessage(f"\n=== MLX export: {xml_path} -> {output_pte} ===")
    run([
        python,
        os.path.join(proj, "bin", "export_mlx.py"),
        xml_path,
        output_pte,
    ], cwd=proj, env=export_env)
    return output_pte


_log_file = None   # set by main() when --log is active


def run(cmd, **kwargs):
    """Run a command, printing it first. Tees output to log file if active.

    With ``_log_file`` set, streams subprocess output line-by-line so
    Python-level ``_Tee`` sees each line. Without it, falls back to a
    plain ``subprocess.run``. Exits the parent process on non-zero return.
    """
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


def _find_lexer(root):
    """Return the XML lexer knob from SymbolicSpace or legacy InputSpace."""
    ss = root.find(".//SymbolicSpace")
    lexer = ss.findtext("lexer") if ss is not None else None
    if lexer:
        return lexer
    inp = root.find(".//InputSpace")
    legacy = inp.findtext("lexer") if inp is not None else None
    if legacy:
        print("[train] DEPRECATED: <lexer> belongs in <SymbolicSpace> "
              "(Phase 4b); found it under <InputSpace>.")
        return legacy
    return None


def read_xml_config(xml_path):
    """Read embedding and training params from the XML config.

    Extracts ``embeddingPath``, the ``SymbolicSpace.lexer`` knob, and the
    ``PerceptualSpace`` ``nDim`` / ``nVectors`` pair if present. Falls back
    to ``data/model.xml`` for the default lexer so configs such as MM_20M
    correctly skip the word-embedding Phase 1 when they inherit raw input.
    Returns a dict that train_local uses to forward consistent flags to
    embed.py.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    arch = root.find("architecture")
    result = {}
    if arch is not None:
        ep = arch.findtext("embeddingPath")
        if ep:
            result["embeddingPath"] = ep
    # Check the lexer knob. Phase 4b (rev. 2026-06-09): <lexer> lives on
    # SymbolicSpace (lexing is analytic cutting); fall back to legacy
    # InputSpace and then to the project default XML.
    lexer = _find_lexer(root)
    if not lexer:
        defaults_path = os.path.join(project_dir(), "data", "model.xml")
        if os.path.exists(defaults_path):
            try:
                lexer = _find_lexer(ET.parse(defaults_path).getroot())
            except ET.ParseError:
                lexer = None
    if lexer:
        result["lexer"] = lexer
    # PerceptualSpace.nDim is the lexicon vector size; SBOW must train at
    # this dim or downstream codebook lookups blow up at load time.
    perc = root.find(".//PerceptualSpace")
    if perc is not None:
        ndim = perc.findtext("nDim")
        if ndim:
            try:
                result["vectorSize"] = int(ndim)
            except ValueError:
                pass
        nvec = perc.findtext("nVectors")
        if nvec:
            try:
                result["nVectors"] = int(nvec)
            except ValueError:
                pass
    return result


def train_local(args):
    """Run training locally: embeddings then model.

    Phase 1 invokes embed.py only when embeddings are absent or
    ``--force-embeddings`` is set (skipped entirely for byte-lexer
    configs). Phase 2 runs Models.py with overrides passed through
    ``BASIC_*`` env vars and respects ``--profile`` for cProfile capture.
    """
    proj = project_dir()
    python = venv_python(proj)
    env = {**os.environ, "PYTHONPATH": os.path.join(proj, "bin"),
           "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
           "PYTHONUNBUFFERED": "1"}
    apply_compile_target_env(args, env)

    # Resolve the XML config path
    xml_path = resolve_model_path(proj, args.model)

    if args.compile_target == "mlx":
        export_mlx_local(args, proj, python, xml_path)
        TheMessage("\n=== Phase 2: Skipped (compile-target=mlx exports a .pte) ===")
        return

    # Read embeddingPath from XML -- output embeddings where Models.py expects them
    cfg = read_xml_config(xml_path)
    emb_relpath = cfg.get("embeddingPath")
    if emb_relpath:
        emb_path = os.path.join(os.path.dirname(xml_path), emb_relpath)
    else:
        emb_path = os.path.join(proj, "output", "embeddings", "sentence.pt")

    # --- Phase 1: Embeddings ---
    if cfg.get("lexer") in ("byte", "bytes", "raw"):
        TheMessage(
            f"\n=== Phase 1: Skipped (lexer={cfg.get('lexer')}, "
            "no word embeddings) ===")
    elif args.force_embeddings or not os.path.exists(emb_path):
        embed_cmd = [
            python, os.path.join(proj, "bin", "embed.py"), "train",
            "--output", emb_path,
        ]
        if args.num_shards is not None:
            embed_cmd += ["--num-shards", str(args.num_shards)]
        if args.max_docs is not None:
            embed_cmd += ["--max-docs", str(args.max_docs)]
        if args.random_shards:
            embed_cmd += ["--random-shards"]
        # Honor PerceptualSpace.nDim from the XML; embed.py defaults to
        # 100, which silently mismatches small-D lexicon configs.
        if cfg.get("vectorSize"):
            embed_cmd += ["--vector-size", str(cfg["vectorSize"])]
        # Forward latent-vector-size when explicitly set; otherwise
        # embed.py applies its own default of max(64, vector_size) so
        # small-D codebooks get the high-dim-then-PCA pipeline by default.
        if args.latent_vector_size is not None:
            embed_cmd += ["--latent-vector-size", str(args.latent_vector_size)]
        if args.embed_lr is not None:
            embed_cmd += ["--learning-rate", str(args.embed_lr)]
        TheMessage(f"\n=== Phase 1: Training embeddings -> {emb_path} ===")
        run(embed_cmd, cwd=proj, env=env)
    else:
        TheMessage(f"\n=== Phase 1: Embeddings exist at {emb_path}, skipping ===")

    # --- Phase 2: Model training ---
    # Pass overrides via env so Models.py respects them over XML config
    model_env = dict(env)
    model_env["PYTHONUNBUFFERED"] = "1"  # flush stdout line-by-line for live logging
    if args.data is not None:
        model_env["BASIC_DATASET"] = args.data
    if args.max_docs is not None:
        model_env["BASIC_MAX_DOCS"] = str(args.max_docs)
    if args.num_shards is not None:
        model_env["BASIC_NUM_SHARDS"] = str(args.num_shards)
    if args.num_epochs is not None:
        model_env["BASIC_NUM_EPOCHS"] = str(args.num_epochs)
    if args.max_tokens is not None:
        model_env["BASIC_MAX_TOKENS"] = str(args.max_tokens)
    if args.batches is not None:
        model_env["BASIC_MAX_BATCHES"] = str(args.batches)
    # --test gating: env var encodes the request to runTrial in Models.py.
    #   unset             -> skip baseline + post-train test passes
    #   "" (empty string) -> run the test passes uncapped
    #   "N" (integer)     -> run the test passes capped at N batches each
    if args.test is not None:
        model_env["BASIC_RUN_TEST"] = "" if args.test == -1 else str(args.test)
    entry = os.path.join(proj, "bin", "Models.py")

    if args.profile:
        # Run under cProfile via -m cProfile -o <file>
        prof_dir = os.path.join(proj, "output", "profiles")
        os.makedirs(prof_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prof_path = os.path.join(prof_dir, f"train_{ts}.prof")
        model_cmd = [
            python, "-m", "cProfile", "-o", prof_path,
            entry, args.model,
        ]
        TheMessage(f"\n=== Phase 2: Training model (profiling -> {prof_path}) ===")
        run(model_cmd, cwd=os.path.join(proj, "bin"), env=model_env)

        # Print summary
        TheMessage(f"\n=== Profile summary (top 30 by cumulative time) ===")
        stats = pstats.Stats(prof_path, stream=sys.stdout)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        TheMessage(f"\nFull profile saved to {prof_path}")
        TheMessage("View interactively:  snakeviz " + prof_path)
    else:
        model_cmd = [python, entry, args.model]
        TheMessage("\n=== Phase 2: Training model ===")
        run(model_cmd, cwd=os.path.join(proj, "bin"), env=model_env)


def train_remote(args):
    """Rsync project to remote host and run training there.

    Bi-directionally reconciles weight files first (so either side's
    newer checkpoint wins), syncs source code (local authoritative for
    non-weights), then SSHes into the host and runs ``bin/train.py``
    with the same flag set. Pulls generated weights back on completion.
    """
    proj = project_dir()
    ssh_cmd_base = ["ssh"]
    if args.key_file:
        ssh_cmd_base += ["-i", os.path.expanduser(args.key_file)]
    ssh_opts = " ".join(shlex.quote(part) for part in ssh_cmd_base)
    dest = f"{args.user}@{args.host}:{args.remote_dir}/"

    # Reconcile weights first so remote training can resume from the newest
    # checkpoint, regardless of which side produced it.
    TheMessage(f"\n=== Syncing weights with {args.host} ===")
    weight_cmd = [
        "rsync", "-av", "--progress", "--update", "--prune-empty-dirs",
        "-e", ssh_opts,
        "--include", "*/",
    ]
    for pattern in WEIGHT_PATTERNS:
        weight_cmd += ["--include", pattern]
    weight_cmd += ["--exclude", "*"]
    run(weight_cmd + [f"{proj}/", dest])
    run(weight_cmd + [dest, f"{proj}/"])

    # Rsync source after weight reconciliation; local source is authoritative
    # for non-weight files.
    TheMessage(f"\n=== Syncing to {args.host} ===")
    rsync_cmd = [
        "rsync", "-av", "--progress",
        "-e", ssh_opts,
        "--exclude", ".venv",
        "--exclude", "__pycache__",
        "--exclude", ".DS_Store",
        "--exclude", "output/",
    ]
    for pattern in WEIGHT_PATTERNS:
        rsync_cmd += ["--exclude", pattern]
    rsync_cmd += [f"{proj}/", dest]
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
    if args.max_tokens is not None:
        remote_args += ["--max-tokens", str(args.max_tokens)]
    if args.test is not None:
        if args.test == -1:
            remote_args += ["--test"]
        else:
            remote_args += ["--test", str(args.test)]
    if args.compile_target is not None:
        remote_args += ["--compile-target", args.compile_target]
    if args.compile_mode is not None:
        remote_args += ["--compile-mode", args.compile_mode]
    if args.mlx_output is not None:
        remote_args += ["--mlx-output", args.mlx_output]
    if args.random_shards:
        remote_args += ["--random-shards"]
    if args.profile:
        remote_args += ["--profile"]
    # SSH and run -- forward selected env vars that affect training behaviour
    remote_env_vars = "PYTHONUNBUFFERED=1 PYTHONPATH=bin"
    for var in ("BASICMODEL_DEVICE", "MODEL_COMPILE", "MODEL_COMPILE_MODE",
                "BASIC_COMPILE_MODE"):
        val = os.environ.get(var)
        if val:
            remote_env_vars = f"{var}={val} {remote_env_vars}"

    TheMessage(f"\n=== Running training on {args.host} ===")
    remote_cmd = (
        f"cd {args.remote_dir} && "
        f"py=.venv/bin/python; [ -x \"$py\" ] || py=.venv/Scripts/python.exe; "
        f"{remote_env_vars} \"$py\" {' '.join(remote_args)}"
    )
    ssh_cmd = ssh_cmd_base + [f"{args.user}@{args.host}", remote_cmd]
    run(ssh_cmd)

    TheMessage(f"\n=== Pulling generated weights from {args.host} ===")
    run(weight_cmd + [dest, f"{proj}/"])


class _Tee:
    """Write to both an original stream and a log file.

    Tiny stream wrapper used by ``--log`` to mirror stdout / stderr
    into a transcript file. Flushes the log on every write so a crash
    leaves a complete record on disk.
    """
    def __init__(self, original, log):
        """Bind the original stream and the open log file handle."""
        self._original = original
        self._log = log

    def write(self, data):
        """Write to both the original stream and the log; flush the log."""
        self._original.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        """Flush both underlying streams."""
        self._original.flush()
        self._log.flush()

    def fileno(self):
        """Return the original stream's fileno so isatty/redirect work."""
        return self._original.fileno()


def main():
    """CLI entry point: set up logging tee, time the run, dispatch.

    Opens the ``--log`` transcript file (autogenerated name when ``auto``),
    swaps ``sys.stdout`` / ``sys.stderr`` for ``_Tee``, prints start /
    end timestamps, and routes to ``train_remote`` when ``--host`` is
    set or ``train_local`` otherwise.
    """
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

    start_time = datetime.datetime.now()
    print(f"[train] start: {start_time.isoformat(timespec='seconds')}", flush=True)

    try:
        if args.host:
            train_remote(args)
        else:
            train_local(args)
    finally:
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time
        print(
            f"[train] end:   {end_time.isoformat(timespec='seconds')} "
            f"(elapsed {elapsed})",
            flush=True,
        )
        if _log_file:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            _log_file.close()


if __name__ == "__main__":
    main()
