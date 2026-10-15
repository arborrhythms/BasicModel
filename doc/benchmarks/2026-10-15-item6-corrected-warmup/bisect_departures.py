"""Paired source-boundary replay of the three unchanged native certificates.

The only production delta from accepted R2 to the reviewed landing is the
search exemption. Select either exact forward_discard implementation while
holding all other code and diagnostic initialization fixed. The real test
assertions, reservoirs, costs, comparisons and optimizer remain in place.
"""
import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import traceback
import warnings

import pytest
import torch
import numpy as np

from identity_experiment import save

TESTS = dict(
    bound=('test_math_chain_section14', 'test_forced_ordinary_bound_declaratives_open_no_episode'),
    fork=('test_math_chain_repair2', 'test_eight_corpus_sentences_train_at_distinct_fork_rounds'),
    answer=('test_math_chain_section14', 'test_forced_ordinary_answer_fills_committed_question_without_its_own_episode'))


def serial(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('certificate', choices=TESTS)
    parser.add_argument('charge', choices=('old', 'new'))
    parser.add_argument('out', type=Path)
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    import util, SentenceCredit
    from SentenceFork import SentenceFork
    from reading_fixtures import use_eager_reading
    util.init_device('cpu')
    old_source = subprocess.check_output(['git', 'show', '90ef40a42:bin/SentenceCredit.py'], text=True)
    namespace = {}
    exec(compile(old_source, '90ef40a42:bin/SentenceCredit.py', 'exec'), namespace)
    functions = dict(old=namespace['forward_discard'], new=SentenceCredit.forward_discard)
    selected = functions[args.charge]
    other = functions['new' if args.charge == 'old' else 'old']
    costs, draws, comparisons, reservoirs, matches, associations = [], [], [], [], [], []
    current_candidates = defaultdict(list)
    current_takes = defaultdict(list)
    record = SentenceFork.record
    departure, comparison = SentenceCredit.departure, SentenceCredit.comparison

    def charged(*values, **kwargs):
        result = selected(*values, **kwargs)
        with torch.no_grad():
            alternate = other(*values, **kwargs)
        frames = [p.operation_values for p in values[1] if p is not None and p.operation_values is not None]
        costs.append(dict(selected=serial(result), alternate=serial(alternate),
                          meaning_start=int(values[2]), content_width=int(frames[0].shape[-1]) if frames else None,
                          input_meaning_mass=sum(float(f[:, :2, values[2]:].detach().abs().sum()) for f in frames),
                          maximum_difference=float((result.detach()-alternate).abs().max())))
        return result

    def recorded(fork, choice, slot, state, **kwargs):
        eligible = fork.current_rows & choice.valid & (choice.alternative_count > 0)
        if not fork.explore:
            for row in eligible.nonzero().flatten().tolist():
                current_candidates[row].append(int(slot[row]) if torch.is_tensor(slot) and slot.numel() > 1 else int(slot))
        before = fork.slot.clone()
        result = record(fork, choice, slot, state, **kwargs)
        if not fork.explore:
            for row in (fork.slot != before).nonzero().flatten().tolist():
                current_takes[row].append(int(fork.slot[row]))
        return result

    def departed(*values, **kwargs):
        result = departure(*values, **kwargs)
        draws.append(serial(result))
        reservoirs.append(dict(eligible=dict(current_candidates), replacements=dict(current_takes)))
        current_candidates.clear(); current_takes.clear()
        return result

    def compared(*values, **kwargs):
        result = comparison(*values, **kwargs)
        comparisons.append(serial(result))
        return result

    import BindingAnswers
    from ConceptIndex import ClauseTaxonomyPlan
    native_particular = ClauseTaxonomyPlan.bind_particular
    def associated(plan, clause, row_id):
        from ThoughtReferences import open_slots
        sources = {clause.refs[0]} if type(clause.refs[0]) is int else set()
        if clause.subject_word_id > 0:
            sources.add(clause.subject_word_id)
        before = {form: sorted(ids.intersection(sources)) for form, ids in plan.alloc.word_forms.items()
                  if ids.intersection(sources)}
        result = native_particular(plan, clause, row_id)
        associations.append(dict(row_id=int(row_id), slots=int(clause.slots.shape[0]),
            refs=[ref if type(ref) is int else repr(ref) for ref in clause.refs],
            subject_word_id=int(clause.subject_word_id), open=serial(open_slots(clause.meaning)),
            source_forms=before,
            inherited_forms=sorted(form for form, ids in plan.alloc.word_forms.items() if row_id in ids)))
        return result

    native_matches = BindingAnswers.matches
    def matched(model, value, target):
        result = native_matches(model, value, target)
        from ThoughtReferences import open_slots, bindings
        metadata = bindings(value)
        chain = []
        current = value
        references = BindingAnswers.references(value)
        reference = references[0][0] if len(references) == 1 else None
        seen = set()
        store = model.symbolSpace.ltm_store
        while reference is not None and reference not in seen and len(chain) < 64:
            seen.add(reference)
            chain.append(dict(reference=serial(reference)))
            if reference[0] == 'sym':
                break
            if reference[0] == 'ltm':
                index = store._index_occurrences.get(reference)
                if index is None:
                    chain[-1]['missing'] = True
                    break
                chain[-1].update(row_id=int(store.row_ids[index]), physical_refs=serial(store.refs[index]),
                    kind=store.KINDS[int(store.record_kind[index])], order=int(store.order[index]))
                current = store.meaning_of(index)
            elif reference[0] == 'constituent':
                current = current.constituents[reference[1]]
            else:
                break
            if current is None:
                break
            chain[-1].update(role_mask=serial(current.role_mask), role_refs=serial(current.role_refs),
                                role_norms=serial(current.roles.norm(dim=-1)))
            if int(current.role_mask.sum()) != 1:
                break
            reference = current.role_refs[int(current.role_mask.nonzero()[0])]
        matches.append(dict(target=target, correct=result, open=serial(open_slots(value)),
                            role_refs=serial(value.role_refs),
                            known=list(model._concept_owner().word_concepts(target)), chain=chain,
                            bindings={key:serial(metadata.get(key)) for key in
                                ('_forward_references', '_bound_roles', '_open_references')}))
        return result

    error = None
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        use_eager_reading(patch)
        patch.setenv('BASIC_SEED', str(args.seed))
        patch.setattr(SentenceCredit, 'forward_discard', charged)
        patch.setattr(SentenceFork, 'record', recorded)
        patch.setattr(SentenceCredit, 'departure', departed)
        patch.setattr(SentenceCredit, 'comparison', compared)
        patch.setattr(BindingAnswers, 'matches', matched)
        patch.setattr(ClauseTaxonomyPlan, 'bind_particular', associated)
        module, name = TESTS[args.certificate]
        certificate = getattr(importlib.import_module(module), name)
        # These fixtures call from_config directly, bypassing the training
        # entry point that reads BASIC_SEED. Pin the replay RNGs explicitly.
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            certificate(args.out, None)
        except Exception:
            error = traceback.format_exc()
    result = dict(certificate=args.certificate, charge=args.charge, seed=args.seed,
        source_before='90ef40a42', source_after='f1dbd12a2', passed=error is None, error=error,
        forward_charges=costs, departures=draws, reservoirs=reservoirs,
        comparisons=comparisons, answers=matches, lexical_associations=associations)
    save(args.out/'result.json', result)
    print(json.dumps(dict(certificate=args.certificate, charge=args.charge, passed=error is None,
        charge_calls=len(costs), changed_charge_calls=sum(row['maximum_difference'] > 0 for row in costs),
        compose_rounds=[draw['compose_round'] for draw in draws], error=error)), flush=True)


if __name__ == '__main__':
    main()
