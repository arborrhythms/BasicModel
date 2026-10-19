"""Lesson artifacts are explicit prerequisites, never invitations to train.

Paths in the registry are relative to the repository root. A private registry
may be selected with BASICMODEL_PHASE_ARTIFACTS; it replaces the entire map.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'test/fixtures/phase-artifacts.json'


@dataclass(frozen=True)
class PhaseArtifact:
    path: str
    sha256: str
    receipt: str
    config: str | None = None
    minimum_sentences: int | None = None

    def __post_init__(self):
        if not self.path or not self.receipt or not re.fullmatch('[0-9a-f]{64}', self.sha256):
            raise ValueError('phase artifact requires a path, SHA-256 and producing receipt')


class PhaseRegistry:
    def __init__(self, phases, root=ROOT):
        self.phases, self.root = dict(phases), Path(root)

    @classmethod
    def read(cls, path=REGISTRY, root=ROOT):
        data = json.loads(Path(path).read_text())
        if data['schema'] != 1:
            raise ValueError('unsupported phase-artifact registry schema')
        return cls({phase: None if entry is None else PhaseArtifact(**entry)
                    for phase, entry in data['phases'].items()}, root)

    def write(self, path):
        Path(path).write_text(json.dumps(dict(schema=1, phases={
            phase: None if entry is None else {key: value for key, value in asdict(entry).items()
                                               if value is not None}
            for phase, entry in self.phases.items()}), indent=2) + '\n')

    def resolve(self, path):
        return (self.root / path).resolve()

    def require(self, phase):
        if phase not in self.phases:
            pytest.fail(f'unknown certificate phase: {phase}', pytrace=False)
        artifact = self.phases[phase]
        if artifact is None:
            pytest.skip(f'phase artifact absent: {phase} (no producing lesson registered)')
        path = self.resolve(artifact.path)
        if not path.is_file():
            pytest.skip(f'phase artifact absent: {phase} ({artifact.path})')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != artifact.sha256:
            pytest.fail(f'phase artifact hash mismatch: {phase}: expected {artifact.sha256}, '
                        f'got {digest} ({artifact.path})', pytrace=False)
        if not self.resolve(artifact.receipt).is_file():
            pytest.fail(f'phase producing receipt absent: {phase}: {artifact.receipt}', pytrace=False)
        return artifact


def certificate_phase(item):
    marker = item.get_closest_marker('phase')
    if marker is None:
        return None
    if len(marker.args) != 1 or marker.kwargs or not isinstance(marker.args[0], str):
        raise ValueError('phase marker requires one lesson phase name')
    return marker.args[0]


@pytest.fixture
def phase_registry():
    return PhaseRegistry.read(os.environ.get('BASICMODEL_PHASE_ARTIFACTS', REGISTRY))


@pytest.fixture
def phase_artifact(request, phase_registry):
    phase = certificate_phase(request.node)
    if phase is None:
        pytest.fail('artifact restoration requires a phase marker', pytrace=False)
    return phase_registry.require(phase)


@pytest.fixture(autouse=True)
def _phase_prerequisite(request):
    if certificate_phase(request.node) is not None:
        request.getfixturevalue('phase_artifact')


@pytest.fixture
def restore_phase(phase_artifact, phase_registry):
    """Restore through the consumer's loader, after the exact artifact gate.

    A rejected restore fails; there is no fresh-model or training fallback.
    """
    def restore(loader):
        result = loader(phase_registry.resolve(phase_artifact.path))
        if result is False:
            pytest.fail('registered phase artifact could not be restored', pytrace=False)
        return result
    return restore
