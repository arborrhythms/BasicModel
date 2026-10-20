"""Four unseeded diagnostic initializations of the unchanged pending certificate."""
from pathlib import Path
import json
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
from reading_fixtures import use_eager_reading
from ThoughtReferences import bindings, open_slots
import BindingAnswers
from test_math_chain_section14 import test_forced_ordinary_pending_premise_in_both_orders


def main():
    destination = Path(__file__).parent/'pending-population'
    destination.mkdir(exist_ok=True)
    outcomes = []
    for attempt in range(4):
        folder = destination/str(attempt)
        folder.mkdir(exist_ok=True)
        # Diagnostic replay may restore this state. No test chooses a seed.
        torch.save(dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                        python=random.getstate()), folder/'initial-rng.pt')
        patch = pytest.MonkeyPatch()
        use_eager_reading(patch)
        original = BindingAnswers.matches
        observations = []
        def describe(meaning):
            return dict(refs=meaning.role_refs, open=open_slots(meaning),
                        mask=meaning.role_mask.tolist(), bindings=bindings(meaning))
        def probe(model, meaning, word):
            answer = original(model, meaning, word)
            store = model.symbolSpace.ltm_store
            observations.append(dict(target=word, matched=answer,
                known=model._concept_owner().word_concepts(word), value=describe(meaning),
                rows=[dict(index=i, id=int(store.row_ids[i]),
                    kind=store.KINDS[int(store.record_kind[i])],
                    rawrefs=store.refs[i].tolist(), value=describe(store.meaning_of(i)))
                    for i in range(len(store))
                    if store.KINDS[int(store.record_kind[i])] != 'estimate']))
            (folder/'bindings.json').write_text(json.dumps(observations, default=str, indent=2)+'\n')
            return answer
        patch.setattr(BindingAnswers, 'matches', probe)
        try:
            test_forced_ordinary_pending_premise_in_both_orders(folder, False, None)
            outcomes.append(dict(attempt=attempt, outcome='passed'))
        except AssertionError as error:
            outcomes.append(dict(attempt=attempt, outcome='failed', reason=str(error)))
        finally:
            patch.undo()
        (destination/'outcomes.json').write_text(json.dumps(outcomes, indent=2)+'\n')
        print(outcomes[-1], flush=True)


if __name__ == '__main__':
    main()
