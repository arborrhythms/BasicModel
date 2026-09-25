"""Preserve the accepted-review host-sync correction and its exact source."""
import argparse
from collections import Counter
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


def package(label, source):
    run = ROOT / "output" / f"item9-bank-sync-{label}"
    result = json.loads((run / "result.json").read_text())
    manifest = json.loads((run / "source-manifest.json").read_text())
    measured = manifest["validated_source"]
    assert Counter(result["selected"]) == Counter(result["completed"])
    assert len(result["selected"]) == len(set(result["selected"]))
    assert result["exit_code"] == (1 if label == "red" else 0), (label, result["reason"])
    if label != "red":
        assert measured == source, label
    reports = [r for p in run.glob("worker-*.json")
               for r in json.loads(p.read_text()).get("reports", [])]
    counts = Counter()
    for node in result["selected"]:
        found = [r for r in reports if r["nodeid"] == node]
        calls = [r for r in found if r["phase"] == "call"]
        other = [r for r in found if r["outcome"] in ("skipped", "xfailed", "failed")]
        final = (calls or other)[-1]
        if label != "red":
            assert final["outcome"] not in ("failed", "xpassed"), final
        counts[final["outcome"]] += 1
    summary = {k: result[k] for k in ("reason", "exit_code", "elapsed_seconds", "limits",
                                    "peak_aggregate_memory_bytes", "compile_cache_retries")}
    summary.update(selected=len(result["selected"]), completed=len(result["completed"]),
                   outcomes=dict(counts), source_files=len(measured),
                   source_sha256=hashlib.sha256(json.dumps(measured, sort_keys=True).encode()).hexdigest())
    write(HERE / f"{label}-summary.json", summary)
    shutil.copyfile(run / "source-manifest.json", HERE / f"{label}-source-manifest.json")
    (HERE / f"{label}-result.json.gz").write_bytes(gzip.compress((run / "result.json").read_bytes(), mtime=0))
    logs = b"".join(p.name.encode() + b"\n" + p.read_bytes() for p in sorted(run.glob("worker-*.log")))
    (HERE / f"{label}-workers.log.gz").write_bytes(gzip.compress(logs, mtime=0))
    return summary


def main(before_doc_links):
    source = source_snapshot(ROOT)
    labels = ["red", "affected", "slow", "full"]
    if not before_doc_links:
        labels.append("doc-links")
    checks = {label: package(label, source) for label in labels}
    diagnostics = HERE / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for label in ("red-initial", "affected-initial", "slow-initial", "focused"):
        for p in (ROOT / "output" / f"item9-bank-sync-{label}").glob("*"):
            if p.suffix in (".json", ".log"):
                (diagnostics / (label + "-" + p.name + ".gz")).write_bytes(
                    gzip.compress(p.read_bytes(), mtime=0))
    run = ROOT / "output/item9-bank-sync-measurements"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["source"] == source and manifest["source_unchanged"]
    assert len(manifest["completed"]) == 4
    assert all(r["exit_code"] == 0 for r in manifest["completed"])
    assert json.loads((run / "comparison.json").read_text())["parity_demonstrated"]
    baseline = json.loads((run / "baseline.json").read_text())
    previous_baseline = json.loads((HERE.parent / "2026-09-25-item9-bank/measurements/baseline.json").read_text())
    assert len(baseline["phases"]) == len(previous_baseline["phases"])
    for before, after in zip(previous_baseline["phases"], baseline["phases"]):
        assert before["name"] == after["name"]
        assert before["reconstruction_mean"] == after["reconstruction_mean"]
        assert [(s["reconstruction"], s["answer"]) for s in before["steps"]] == [
            (s["reconstruction"], s["answer"]) for s in after["steps"]]
    write(HERE / "baseline-comparison.json", dict(
        preceding_receipt="../2026-09-25-item9-bank/measurements/baseline.json",
        all_reconstruction_and_answer_steps_identical=True,
        phase_means={phase["name"]: phase["reconstruction_mean"] for phase in baseline["phases"]}))
    destination = HERE / "measurements"
    destination.mkdir(exist_ok=True)
    for p in run.iterdir():
        if p.suffix == ".log":
            (destination / (p.name + ".gz")).write_bytes(gzip.compress(p.read_bytes(), mtime=0))
        elif p.is_file():
            shutil.copyfile(p, destination / p.name)
    write(HERE / "review-source.json", source)
    previous = json.loads((HERE.parent / "2026-09-25-item9-bank/review-source.json").read_text())
    write(HERE / "source-delta.json", {name: dict(before=previous.get(name), after=source.get(name))
          for name in sorted(set(source) | set(previous)) if source.get(name) != previous.get(name)})
    with tarfile.open(HERE / "review-source.tar.gz", "w:gz") as archive:
        for name in sorted(source):
            info = archive.gettarinfo(ROOT / name, arcname=name)
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ""
            with (ROOT / name).open("rb") as f:
                archive.addfile(info, f)
    (HERE / "tracked-source.patch").write_bytes(subprocess.check_output(
        ["git", "diff", "--", "bin", "test", "data"], cwd=ROOT))
    write(HERE / "validation-summary.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-doc-links", action="store_true")
    main(parser.parse_args().before_doc_links)
