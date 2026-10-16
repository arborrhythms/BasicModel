"""Coverage of native departures on a trained artifact's frozen greedy menus.

This samples a run at fixed learned parameters. It does not promise coverage
in one batch or across menus changed by later optimizer steps. For E eligible
decisions and at most W walks, a particular decision is selected with at least
1/(W*E) probability per presentation. R >= max(W*E)*log(N/1e-6)
presentations give a union bound <= 1e-6 for missing any of N decisions.
"""
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.artifact_eval
def test_trained_artifact_departs_every_eligible_sentence_round(tmp_path, monkeypatch):
    artifact = os.environ.get('BASICMODEL_EXPLORATION_ARTIFACT')
    if not artifact:
        pytest.skip('requires BASICMODEL_EXPLORATION_ARTIFACT from paired_identity_experiment')
    artifact = Path(artifact)
    metadata = json.loads((artifact/'singletons.json').read_text())
    assert metadata['optimizer_steps'] > 0 and metadata['completed']
    from identity_experiment import make_model, restore_warmup, save
    from Models import BaseModel
    from SentenceFork import SentenceFork
    import SentenceCredit
    departure = SentenceCredit.departure
    restore = BaseModel.load_weights
    torch.set_num_threads(1)
    model = make_model(tmp_path, monkeypatch)
    menus, observed = defaultdict(list), {}
    record = SentenceFork.record

    class Captured(Exception):
        pass

    def recorded(fork, choice, slot, state, **kwargs):
        eligible = fork.current_rows & choice.valid & (choice.alternative_count > 0)
        if not fork.explore:
            slots = torch.as_tensor(slot).expand_as(fork.slot)
            for row in eligible.nonzero().flatten().tolist():
                menus[row].append(int(slots[row]))
        return record(fork, choice, slot, state, **kwargs)

    def capture(narrowing, compose, **kwargs):
        attention = (compose.new_zeros(compose.shape[0], 0) if narrowing is None
                     else narrowing.alternatives.clone())
        if narrowing is not None and kwargs.get('sentence_ids') is not None:
            owners = kwargs['sentence_ids'].gather(1, narrowing.round_words.clamp_min(0))
            attention &= (owners == kwargs['sentence']) & (narrowing.round_words >= 0)
        observed.update(attention=attention.detach(), compose=compose.detach(),
                        candidates=kwargs.get('candidates'), active=kwargs['active'].detach())
        # Both trial costs precede optimizer updates. This stops before the
        # explore trial, update or commit; only its native menu is measured.
        raise Captured

    fields = ['the black cat runs .', 'the white dog sleeps .',
              'a cat sees a dog .', 'a dog sees a cat .']
    try:
        restore_warmup(model, artifact/'warmup.ckpt', restore)
        monkeypatch.setattr(SentenceFork, 'record', recorded)
        monkeypatch.setattr(SentenceCredit, 'departure', capture)
        from attention_curriculum_gates import present
        with pytest.raises(Captured):
            present(model, dict(fields=fields), train=True, optimizer=model.getOptimizer(lr=.003))
    finally:
        model.End()
    monkeypatch.setattr(SentenceFork, 'record', record)
    monkeypatch.setattr(SentenceCredit, 'departure', departure)
    # Use the production selection functions on the artifact's actual menu,
    # with independent rows for the bounded repeated presentations.
    counts = [len(menus[b]) for b in range(len(fields))]
    assert all(counts), 'every declared sentence must have an eligible compose round'
    candidates = observed['candidates']
    walks = (observed['attention'].any(-1).long() + observed['compose'].any(-1).long()
             + (0 if candidates is None else candidates.any(-1).long()))
    total = sum(counts)
    repetitions = math.ceil(max(c * int(w) for c, w in zip(counts, walks)) * math.log(total/1e-6))
    torch.manual_seed(20261015)
    active = observed['active'].repeat(repetitions)
    fork = SentenceFork(active)
    dummy = torch.zeros(len(active), 1)
    for index in range(max(counts)):
        slots = torch.tensor([menus[b][index] if index < counts[b] else -1
                              for b in range(len(fields))]).repeat(repetitions)
        valid = slots >= 0
        choice = SimpleNamespace(valid=valid, alternative_count=valid.long(), action=slots.clamp_min(0))
        fork.record(choice, slots, dummy, word=torch.zeros_like(slots), latches=(dummy,))
    draw = departure(SimpleNamespace(alternatives=observed['attention'].repeat(repetitions, 1)),
        observed['compose'].repeat(repetitions, 1), active=active, compose_round=fork.slot,
        candidates=None if candidates is None else candidates.repeat(repetitions, 1))
    selected = draw['compose_round'].reshape(repetitions, len(fields))
    rows = []
    for b, field in enumerate(fields):
        eligible, seen = set(menus[b]), set(selected[:, b].tolist()) - {-1}
        rows.append(dict(sentence=field, eligible=sorted(eligible), departed=sorted(seen),
                         missing=sorted(eligible-seen)))
    save(tmp_path/'coverage.json', dict(repetitions=repetitions, union_bound=1e-6,
        seed=20261015, fixed_learned_menus=True, rows=rows))
    assert all(not row['missing'] for row in rows), rows
