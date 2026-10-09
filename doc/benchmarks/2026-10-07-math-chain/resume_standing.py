"""Continue startup-only failures from their exact saved entropy entry states.

The ten sum launchers failed importing an audit dependency before constructing
a model or running an epoch. Their original folders remain intact. This
supplies that dependency and preserves every initial state; it does not retry
a training or select an initialization by its result.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    import campaign
    if len(sys.argv) > 1:
        import random
        import numpy as np
        import torch
        import rng_replay
        original = rng_replay.entry
        def entry(folder):
            saved = HERE / 'standing-launch-failure' / folder.name / 'unseeded-entry.pt'
            state = torch.load(saved, map_location='cpu', weights_only=False)
            random.setstate(state['python'])
            np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch'])
            original(folder)
            (folder / 'startup-continuation.json').write_text(json.dumps(dict(
                original_entry=str(saved.relative_to(HERE)),
                original_entry_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
                reason='missing audit import before model construction',
                prior_training_epochs=0, fresh_initialization_draws=0)) + '\n')
        rng_replay.entry = entry
        campaign.sum_child(Path(sys.argv[2]))
        return
    supplemental = json.loads((HERE / 'measurement-supplement.json').read_text())['files']
    snapshot = campaign.bounded.source_snapshot
    def checked_snapshot(root):
        assert all(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
                   for name, digest in supplemental.items())
        return snapshot(root)
    campaign.bounded.source_snapshot = checked_snapshot
    # Reuse the declared campaign unchanged; sum children enter this wrapper.
    campaign.__file__ = str(Path(__file__).resolve())
    campaign.campaign()


if __name__ == '__main__':
    main()
