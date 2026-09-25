"""Verify every reviewed source blob after the authorized implementation commit."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "test"))
from bounded_tests import source_snapshot

source = json.loads((HERE / "review-source.json").read_text())
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT).decode().strip()
names = sorted(source)
requests = "".join(f"{commit}:{name}\n" for name in names).encode()
result = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT, input=requests,
                        stdout=subprocess.PIPE, check=True)
stream = io.BytesIO(result.stdout)
mismatches = []
for name in names:
    header = stream.readline().decode().split()
    assert len(header) == 3 and header[1] == "blob", (name, header)
    data = stream.read(int(header[2]))
    assert stream.read(1) == b"\n"
    actual = hashlib.sha256(data).hexdigest()
    if actual != source[name]:
        mismatches.append(dict(path=name, reviewed=source[name], committed=actual))
assert not stream.read()
assert not mismatches, mismatches
assert source_snapshot(ROOT) == source
verification = dict(
    implementation_commit=commit, implementation_tree=tree,
    reviewed_source_files=len(source), committed_source_files_verified=len(source),
    source_map_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
    committed_blobs_match_reviewed_source=True, working_source_matches_reviewed_source=True,
    mismatches=mismatches,
    method="SHA-256 of every committed source blob read with git cat-file --batch, compared with review-source.json.",
    validation_summary="validation-summary.json",
    publication_approval="Alec, September 25, 2026: Then commit and push all changes in the tree, and begin the next item (now called 9b).",
)
(HERE / "committed-source-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
print(json.dumps(verification, indent=2))
