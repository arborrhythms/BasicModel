"""Package source-matched bank-contract review checks without rewriting history."""
from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "test"))
from bounded_tests import source_snapshot


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def receipt(label, source):
    run = ROOT / "output" / f"item9-bank-{'full-final' if label == 'full' else label}"
    result = json.loads((run / "result.json").read_text())
    manifest = json.loads((run / "source-manifest.json").read_text())
    assert result["exit_code"] == 0, (label, result["reason"])
    assert len(result["selected"]) == len(set(result["selected"]))
    assert Counter(result["selected"]) == Counter(result["completed"])
    measured = manifest["validated_source"]
    delta = source_delta(measured, source)
    if label in ("affected", "slow"):
        assert set(delta) <= {"test/test_reconstruction_bank_contract.py"}, (label, delta)
    else:
        assert not delta, (label, delta)
    reports = [r for p in sorted(run.glob("worker-*.json"))
               for r in json.loads(p.read_text()).get("reports", [])]
    counts = Counter()
    for node in result["selected"]:
        found = [r for r in reports if r["nodeid"] == node]
        calls = [r for r in found if r["phase"] == "call"]
        exceptions = [r for r in found if r["outcome"] in ("failed", "skipped", "xfailed")]
        final = (calls or exceptions)[-1]
        assert final["outcome"] not in ("failed", "xpassed"), final
        counts[final["outcome"]] += 1
    summary = {k: result[k] for k in ("reason", "exit_code", "elapsed_seconds", "limits",
                                    "peak_aggregate_memory_bytes", "compile_cache_retries")}
    summary.update(selected=len(result["selected"]), completed=len(result["completed"]),
                   outcomes=dict(counts), source_files=len(measured), source_delta_from_review=delta,
                   source_sha256=hashlib.sha256(json.dumps(measured, sort_keys=True).encode()).hexdigest())
    write(HERE / f"{label}-summary.json", summary)
    (HERE / f"{label}-result.json.gz").write_bytes(gzip.compress((run / "result.json").read_bytes(), mtime=0))
    shutil.copyfile(run / "source-manifest.json", HERE / f"{label}-source-manifest.json")
    logs = b"".join(p.name.encode() + b"\n" + p.read_bytes() for p in sorted(run.glob("worker-*.log")))
    (HERE / f"{label}-workers.log.gz").write_bytes(gzip.compress(logs, mtime=0))
    return summary


def source_delta(before, after):
    return {name: dict(before=before.get(name), after=after.get(name))
            for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)}


def main(*, before_doc_links=False):
    source = source_snapshot(ROOT)
    labels = ("affected", "slow", "cleanup", "full")
    if not before_doc_links:
        labels += ("doc-links",)
    checks = {label: receipt(label, source) for label in labels}
    run = ROOT / "output/item9-bank-measurements"
    manifest = json.loads((run / "manifest.json").read_text())
    measurement_delta = source_delta(manifest["source"], source)
    assert set(measurement_delta) <= {"test/test_reconstruction_bank_contract.py"}
    assert manifest["source_unchanged"]
    write(HERE / "measurement-source-delta.json", measurement_delta)
    assert len(manifest["completed"]) == 4 and all(r["exit_code"] == 0 for r in manifest["completed"])
    assert json.loads((run / "comparison.json").read_text())["parity_demonstrated"]
    destination = HERE / "measurements"
    destination.mkdir(exist_ok=True)
    for p in run.iterdir():
        if p.suffix == ".log":
            (destination / (p.name + ".gz")).write_bytes(gzip.compress(p.read_bytes(), mtime=0))
        elif p.is_file():
            shutil.copyfile(p, destination / p.name)
    diagnostics = HERE / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for name in ("red", "first-fix", "full"):
        for p in (ROOT / "output" / f"item9-bank-{name}").glob("*"):
            if p.suffix in (".json", ".log"):
                (diagnostics / (name + "-" + p.name + ".gz")).write_bytes(gzip.compress(p.read_bytes(), mtime=0))
    probe = HERE / "fallback-probe"
    probe.mkdir(exist_ok=True)
    for p in (ROOT / "output/item9-bank-fallback-probe").iterdir():
        if p.suffix == ".log":
            (probe / (p.name + ".gz")).write_bytes(gzip.compress(p.read_bytes(), mtime=0))
        elif p.is_file():
            shutil.copyfile(p, probe / p.name)
    probe_delta = source_delta(json.loads((probe / "after.json").read_text())["source"], source)
    assert set(probe_delta) <= {"test/test_reconstruction_bank_contract.py"}
    write(probe / "source-delta.json", probe_delta)
    assert all(json.loads((probe / (label + "-process.json")).read_text())["exit_code"] == 0
               for label in ("before", "after"))
    previous = json.loads((HERE.parent / "2026-09-25-item9-parity/review-source.json").read_text())
    write(HERE / "review-source.json", source)
    write(HERE / "source-delta.json", {name: dict(before=previous.get(name), after=source.get(name))
          for name in sorted(set(source) | set(previous)) if source.get(name) != previous.get(name)})
    with tarfile.open(HERE / "review-source.tar.gz", "w:gz") as archive:
        for name in sorted(source):
            info = archive.gettarinfo(ROOT / name, arcname=name)
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ""
            with (ROOT / name).open("rb") as f:
                archive.addfile(info, f)
    (HERE / "tracked-source.patch").write_bytes(subprocess.check_output(["git", "diff", "--", "bin", "test", "data"], cwd=ROOT))
    write(HERE / "validation-summary.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-doc-links", action="store_true",
                        help="package artifacts before checking links to those artifacts")
    main(before_doc_links=parser.parse_args().before_doc_links)
