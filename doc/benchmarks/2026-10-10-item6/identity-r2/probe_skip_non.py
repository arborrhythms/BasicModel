"""One supplied native explore departure per retained clause, no learning.

The diagnostic fixes the reservoir at the first meaning-removing `non`
and chooses its legal STOP alternative. Everything after that departure is
the native suffix. No production legality, scoring, or inverse is changed.
The real keep comparison is observed at learning rate zero; state is discarded.
"""
import argparse
import json
from pathlib import Path

import pytest
import torch

import Models
import SentenceCredit
from SentenceCompose import select_rows
from SentenceFork import SentenceFork, detached
from identity_experiment import make_model, present_documents, recovery_data, restore_warmup, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    loader = Models.BaseModel.load_weights
    reports, fixture, departures = [], [], []
    streams, _ = recovery_data()
    with pytest.MonkeyPatch.context() as patch:
        model = make_model(args.out, patch)
        try:
            drift = restore_warmup(model, args.checkpoint, loader)
            owner = model._concept_owner()
            owner._online_learning_frozen = True
            chart = owner.components._ready()
            start = chart.meaning_start
            params = {k: v.detach().clone() for k, v in model.named_parameters()}
            language = model.languageSpace
            non = next(i for i, r in enumerate(language._compose_unary_rules) if r.method_name == 'non')
            original_choice = language.choose_operation
            operands = None
            def choosing(*a, **kw):
                nonlocal operands
                result = original_choice(*a, **kw)
                operands = result[3]
                return result
            patch.setattr(language, 'choose_operation', choosing)
            record = SentenceFork.record
            def recording(fork, choice, slot, state, *, word, latches):
                if fork.explore:
                    return record(fork, choice, slot, state, word=word, latches=latches)
                eligible = fork.current_rows & choice.valid & (choice.alternative_count > 0)
                fork.count += eligible.long()
                # Consume the native reservoir draw, then supply the requested
                # diagnostic point. It is not a random passing seed.
                if bool(eligible.any()):
                    torch.rand_like(fork.count, dtype=torch.float32)
                take = (eligible & (fork.slot < 0) & (choice.kind == 2) & (choice.local_op == non)
                        & operands[:, 0, start:chart.content_width].abs().sum(-1).gt(0))
                if not bool(take.any()):
                    return
                saved, saved_latches = detached(state), detached(tuple(latches))
                fork.state = saved if fork.state is None else select_rows(fork.state, saved, take)
                fork.latches = saved_latches if fork.latches is None else select_rows(fork.latches, saved_latches, take)
                fork.word = torch.where(take, word, fork.word)
                fork.slot = torch.where(take, slot, fork.slot)
                fork.action = torch.where(take, choice.action, fork.action)
            patch.setattr(SentenceFork, 'record', recording)
            original_departure = SentenceCredit.departure
            def departure(narrowing, compose, **kw):
                result = original_departure(narrowing, compose, **kw)
                slots = kw['compose_round']
                assert (slots >= 0).all(), 'each retained row needs a meaning-removing non'
                width = 0 if narrowing is None else narrowing.alternatives.shape[1]
                result.update(round=width+slots, walk=torch.ones_like(slots),
                    compose_round=slots, narrowing=torch.zeros_like(slots, dtype=torch.bool),
                    attention_round=torch.full_like(slots, -1), candidate_round=torch.full_like(slots, -1),
                    walk_rounds=compose.sum(-1))
                return result
            patch.setattr(SentenceCredit, 'departure', departure)
            layer = language._tree_layer(2)
            choose = layer.select_logits
            def selecting(logits, **kw):
                result = choose(logits, **kw)
                mask = kw.get('masked_action')
                if mask is None or not bool((mask >= 0).any()):
                    return result
                rows = mask >= 0
                allowed = kw['departure_eligible'][:, -1] & torch.isfinite(logits[:, -1])
                assert bool((~rows | allowed).all()), 'STOP must be a native legal departure'
                forced = torch.where(rows, logits.shape[-1]-1, result[0])
                departures.append(dict(mask=rows.tolist(), from_action=mask.tolist(), to_action=forced.tolist()))
                return choose(logits, **dict(kw, replay_action=forced))
            patch.setattr(layer, 'select_logits', selecting)
            score = model._sentence_path_cost
            def scoring(state, sid, active):
                result = score(state, sid, active)
                registry = model._sentence_cost_registry
                records = result[2]['entries']
                programs = [dict(actions=p.actions.detach().cpu(),
                    operation_values=p.operation_values.detach().cpu()) for p in records]
                total = registry.total(objective='reconstruction').detach()
                term = registry._value(registry._terms['reconstruction.forward_discard']).detach()
                fixture.append(dict(programs=programs, other_cost=total.cpu()-term.cpu(),
                    keep_cost=total.cpu(), discard=term.cpu()))
                reports.append(dict(trial=model._sentence_trial, keep_cost=total.tolist(), discard=term.tolist(),
                    terms=registry.breakdown(),
                    closed_meaning_norm=result[2]['record'].root[:, start:chart.content_width].norm(dim=-1).tolist(),
                    actions=[p.actions.tolist() for p in records]))
                return result
            patch.setattr(model, '_sentence_path_cost', scoring)
            optimizer = model.getOptimizer(lr=0.)
            present_documents(model, streams['heldout'][:4], optimizer=optimizer)
            changed = [k for k, v in model.named_parameters() if k not in params or not torch.equal(v, params[k])]
            audit = model._last_sentence_credit
            result = dict(protocol=__doc__, checkpoint=str(args.checkpoint),
                native_loader_drift=drift,
                parameter_changes=changed, trials=reports, departures=departures,
                keep_costs=audit['keep_costs'].tolist(), keep_decision=audit['keep_decision'],
                advantage=audit['advantage'].tolist())
            save(args.out/'result.json', result)
            torch.save(dict(trials=fixture, meaning_start=start), args.out/'retained-trials.pt')
            print(json.dumps(result), flush=True)
            assert not changed, changed
            assert all(decision == 'explore' for decision in audit['keep_decision']), audit['keep_decision']
        finally:
            model.End()


if __name__ == '__main__':
    main()
