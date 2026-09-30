"""Two-truths §7.15/21: bounded local predictor measurements, forced parses.

This isolates the existing predictor and clause/reference mechanisms. It is
not a claim about learned English parsing, identity selection, or full-model
generalization. All conditions start from identical weights, use the same
fixed corpus order and 240 updates. No outcome is asserted or selected.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]

import torch
from ClauseJournal import finish_clause
from Layers import ConceptAllocator, InterSentenceLayer, MeaningExpectation, TernaryTruthStore
from ReferenceContext import SituationFrame
from reading_fixtures import resolve_reading_references
from dataclasses import replace
from Understanding import AnswerProgram
from bounded_tests import source_snapshot


def finish_literal_sum(language, entry):
    """Publish the fixed fixture's literal sum while its reading is open.

    The original protocol supplied left + right as its numerical operator.
    Keep those exact values and expose their pre-fusion frame to the journal;
    a completed row never owns the temporary action record.
    """
    values = entry.reference_values if entry.reference_values is not None else entry.leaves
    ids = entry.reference_ids if entry.reference_ids is not None else entry.concept_ids
    zero = torch.zeros_like(values[0])
    point = values[0] + values[1]
    frames = torch.stack((torch.stack((zero, zero, values[0])),
                          torch.stack((zero, zero, values[1])),
                          torch.stack((values[0], values[1], point))))
    refs = torch.tensor([[-1, -1], [-1, -1], ids.tolist()])
    entry = replace(entry, operation_values=frames, operation_refs=refs,
                    end_state=torch.stack((point, zero, zero)))
    return finish_clause(language, entry)


def corpus(references):
    atoms = torch.eye(8)
    allocator = ConceptAllocator()
    points = {allocator.new_concept(): atom for atom in atoms}
    store = TernaryTruthStore(8, capacity=32)
    store.configure_clause_index(allocate=lambda _: allocator.new_concept(), concept_point=points.get)
    rule = SimpleNamespace(method_name='sum', lhs='exist_O1',
        reference_orders=(('I1', 1),), reference_kinds=(('I1', 'particular'),))
    language = SimpleNamespace(_compose_binary_rules=(rule,), _compose_unary_rules=(),
        forward_binary_step=lambda left, right, *_: left + right)

    def program(person, verb):
        leaves = atoms[[person, verb]]
        return AnswerProgram(rows=torch.tensor([person, verb]), word_rows=torch.tensor([person, verb]),
            activations=torch.ones(2), leaves=leaves,
            concept_ids=torch.tensor([person + 1, verb + 1]),
            actions=torch.tensor([[0, -1, 0], [0, -1, 1], [1, 0, -1]]),
            targets=torch.tensor([0, 1, 1]),
            end_state=torch.stack((leaves.sum(0), atoms[0] * 0, atoms[0] * 0)))

    cases = []
    for person in (0, 1):
        for action, consequence, name in ((2, 4, 'runs → tired'), (3, 5, 'rests → rested')):
            source = finish_literal_sum(language, program(person, action))
            row = store.write_clause(source)
            frame = SituationFrame(1, source.meaning.roles, source.meaning.role_mask,
                int(store.row_ids[row]), source.point.detach())
            target = program(person, consequence)
            if references:
                prediction = MeaningExpectation(
                    torch.stack((source.point, atoms[0] * 0, atoms[0] * 0)), torch.ones(3))
                target = resolve_reading_references(language, target, frames=(frame,),
                    prediction=prediction, forced={0: frame.row_id})
            target = finish_literal_sum(language, target)
            cases.append(dict(label=f'lion {person + 1}: {name}', kind='idea',
                source=source.meaning, target=target.meaning))
    return cases


def observe(model, meaning, kind, document):
    model.predict_and_observe_stm_end_state([3], [meaning.roles], layout='infix',
        role_masks=[meaning.role_mask], sentence_kinds=[kind], documents=[document])


def prediction(model, case, document):
    model.begin_document(0, document)
    observe(model, case['source'], case['kind'], document)
    return model.expect_next_meaning()


def evaluate(model, cases):
    predictions, records = [], []
    with torch.no_grad():
        for i, case in enumerate(cases):
            predicted = prediction(model, case, f'eval-{i}')
            residual = (predicted.roles - case['target'].roles).square().mean(-1)
            present = case['target'].role_mask
            content = present.clone(); content[0] = False
            records.append(dict(label=case['label'], role_mse=float(residual[present].mean()),
                content_mse=float(residual[content].mean()),
                relation_probability=float(predicted.kind_logit.sigmoid())))
            predictions.append(predicted.roles.detach())
    result = dict(mean_role_mse=sum(r['role_mse'] for r in records) / len(records), records=records)
    result['predicted_verb_effect_distance'] = float((predictions[0][1] - predictions[1][1]).norm())
    # Force the other individual's held frame (the resting lion) while keeping
    # the external target "tired" unchanged. Measure only predicate content,
    # excluding the subject address from this wrong-identity surprise.
    result['wrong_identity_content_mse'] = float((predictions[3][1] - cases[0]['target'].roles[1]).square().mean())
    result['correct_identity_content_mse'] = records[0]['content_mse']
    result['wrong_identity_content_rise'] = result['wrong_identity_content_mse'] - records[0]['content_mse']
    return result


def train(model, cases, updates):
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for step in range(updates):
        case = cases[step % len(cases)]
        document = f'train-{step}'
        model.begin_document(0, document)
        observe(model, case['source'], case['kind'], document)
        observe(model, case['target'], case['kind'], document)
        loss = model.consume_inter_loss()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.detach_prediction_context()


def main(path):
    from Meaning import ConceptualMeaning
    torch.set_num_threads(1)
    torch.manual_seed(42)
    source = source_snapshot(ROOT)
    initial = InterSentenceLayer(n_symbols=8, max_depth=8, n_dim=8,
        concept_dim=8, expectation_scope='structured')
    result = dict(protocol=__doc__, seed=42, updates_per_condition=240,
        source_manifest=source, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for reference in (False, True):
        model, cases = copy.deepcopy(initial), corpus(reference)
        before = evaluate(model, cases)
        train(model, cases, 240)
        result['references_on' if reference else 'references_off'] = dict(before=before, after=evaluate(model, cases))
    model, cases = copy.deepcopy(initial), corpus(True)
    atoms = torch.eye(8)
    for left, right in ((0, 6), (1, 7)):
        meaning = ConceptualMeaning(torch.stack((atoms[left], atoms[2], atoms[right])), torch.ones(3, dtype=torch.bool),
                                    sentence_kind='relation')
        cases.append(dict(label=f'generic {left}: part', source=meaning, target=meaning, kind='relation'))
    def kind_metrics():
        records = []
        with torch.no_grad():
            for i, case in enumerate(cases):
                logit = prediction(model, case, f'kind-{i}').kind_logit
                label = float(case['kind'] == 'relation')
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, logit.new_tensor(label))
                records.append((float(loss), float((logit > 0) == bool(label))))
        return dict(mean_bce=sum(r[0] for r in records) / len(records), accuracy=sum(r[1] for r in records) / len(records))
    before = kind_metrics()
    train(model, cases, 240)
    result['mixed_kind'] = dict(before=before, after=kind_metrics(), ideas=4, relations=2)
    assert source_snapshot(ROOT) == source, 'source changed during measurement'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('source_manifest', 'protocol')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
