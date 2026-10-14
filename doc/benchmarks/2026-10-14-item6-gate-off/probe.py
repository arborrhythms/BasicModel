"""Frozen diagnostics of stored singletons and one confounded verb reading.

No training, minting, forced parse, source archive or integrity machinery.
Singletons are read from the checkpoint's native observation rows. The trace
replays the original first held-out batch, recording only its first clause.
"""
import argparse
from pathlib import Path
import warnings

import pytest
import torch
from torch.nn import functional as F

import Models
from attention_curriculum_gates import PERCEPTS, INTERRUPTION, NOVEL
from identity_corpus import KINDS, PROPERTIES, VERBS
from identity_experiment import make_model, present_documents, recovery_data, restore_warmup, save


def cosine(left, right):
    if not bool(left.norm() > 0):
        return None, 'zero closed meaning'
    if not bool(right.norm() > 0):
        return None, 'zero word meaning'
    return float(F.cosine_similarity(left, right, dim=0)), None


def singletons(model):
    owner = model._concept_owner()
    component = owner.components
    component.begin_forward()
    chart = component._ready()
    store = owner._closed_clause_store()
    forms = list(dict.fromkeys((*PERCEPTS, *INTERRUPTION, *NOVEL, *KINDS, *PROPERTIES, *VERBS)))
    atoms, references = {}, {}
    for form in forms:
        word = owner.definitions.word(form=form)
        objects = owner.definitions.objects(word)
        if len(objects) != 1:
            raise RuntimeError(f'{form}: expected one retained word concept, found {objects}')
        concept = objects[0]
        row = owner._csw_row_of(concept)
        if row is None:
            raise RuntimeError(f'{form}: word concept {concept} has no retained code')
        atoms[form] = owner.similarity_codebook.lookup_rows(
            torch.tensor([row], device=chart.basis.device))[0].detach()
        references[int(concept)] = form
    rows = []
    for index in range(len(store)):
        if (store.KINDS[int(store.record_kind[index])] != 'observation'
                or int(store.rel_type[index]) != store.REL_NONE
                or int(store.order[index]) != 0):
            continue
        refs = store.refs[index].tolist()
        if refs[0] not in references or refs[1:] != [-1, -1]:
            continue
        meaning = store.meaning_of(index)
        if meaning is None or meaning.role_mask.tolist() != [True, False, False]:
            raise RuntimeError(f'singleton row {index} has unavailable or multiple meanings')
        form = references[refs[0]]
        value = meaning.roles[0, chart.meaning_start:chart.content_width]
        word_value = atoms[form][chart.meaning_start:chart.content_width]
        similarity, reason = cosine(value, word_value)
        rows.append(dict(index=index, row=int(store.row_ids[index]), word=form,
            concept=refs[0], witnesses=int(store.witness_count[index]),
            closed_meaning_norm=float(value.norm()), word_meaning_norm=float(word_value.norm()),
            cosine=similarity, undefined_reason=reason))
    matrix = component.nouns.matrix().detach()
    if len(matrix) != 58:
        raise RuntimeError(f'expected the 58 accepted warm-up columns, found {len(matrix)}')
    events = torch.stack([chart.expand(chart.project(atoms[form])) for form in forms])
    values = (F.normalize(matrix, dim=-1) @ F.normalize(events, dim=-1).T).abs()
    recovery = [forms.index(form) for form in (*KINDS, *PROPERTIES, *VERBS)]
    columns = []
    for identity, row in zip(component.nouns.ids, values):
        best = int(row.argmax())
        best_recovery = recovery[int(row[recovery].argmax())]
        columns.append(dict(column=identity, nearest_atom=forms[best], nearest_abs_cos=float(row[best]),
            nearest_recovery_atom=forms[best_recovery], nearest_recovery_abs_cos=float(row[best_recovery]),
            abs_cos=dict(zip(forms, row.tolist()))))
    return dict(checkpoint='accepted R/S/T warm-up', meaning_slice=[chart.meaning_start, chart.content_width],
        singleton_selection='native observation, order 0, one reference to a stage-1/2 word concept',
        atom_meaning_norms={form: float(value[chart.meaning_start:chart.content_width].norm())
                            for form, value in atoms.items()},
        singleton_rows=rows, missing_singleton_words=sorted(set(forms)-{r['word'] for r in rows}),
        warmup_columns=columns, atom_space='native dictionary event coordinates',
        column_match_threshold=.9)


def trace(model):
    docs = recovery_data()[0]['heldout'][:4]
    observe = model._sentence_observation
    captures = []
    def observed(state, sid, active, **kwargs):
        result = observe(state, sid, active, **kwargs)
        if kwargs.get('admit'):
            chart = model._concept_owner().components._ready()
            start, end = chart.meaning_start, chart.content_width
            program = result['entries'][0]
            if program is None or program.operation_values is None:
                raise RuntimeError('the retained clause has no forward program')
            words = model._tensor_pushed_ideas[0].detach()
            isp = model.inputSpace
            positions = (isp._word_active_mask[0] & (isp._packed_sentence_ids[0] == sid)).nonzero().flatten()
            source_tokens = dict(zip(positions.tolist(), docs[0]['sentences'][0].split(), strict=True))
            read_positions = model._derivation_program(int(sid))[0][0].tolist()
            read_positions = [position for position in read_positions if position >= 0]
            grammar = getattr(isp, '_ar_grammar_object_atoms', None)
            tokens = []
            for token, position in zip(docs[0]['sentences'][0].split(), positions.tolist(), strict=True):
                tokens.append(dict(token=token, position=position,
                    pushed_meaning_norm=float(words[position, start:end].norm()),
                    grammar_meaning_norm=(float(grammar[0, position, start:end].norm())
                        if torch.is_tensor(grammar) else None)))
            steps = []
            for index, (kind, operation, word) in enumerate(program.actions.tolist()):
                rules = (model.languageSpace._compose_binary_rules if kind == 1 else
                         model.languageSpace._compose_unary_rules if kind == 2 else ())
                values = program.operation_values[index, :, start:end].detach()
                steps.append(dict(step=index, kind=kind, operation=operation, word=word,
                    method=rules[operation].method_name if rules else None,
                    # Leaves have no operation journal entry; its padding is
                    # not an observed zero word. Read their native source row.
                    leaf_position=read_positions[word] if kind == 0 else None,
                    leaf_token=source_tokens[read_positions[word]] if kind == 0 else None,
                    leaf_meaning_norm=(float(words[read_positions[word], start:end].norm())
                                       if kind == 0 else None),
                    meaning_norms=values.norm(dim=-1).tolist() if kind else None,
                    meaning_values=values.tolist() if kind else None))
            methods = [step['method'] for step in steps if step['method']]
            if methods != ['verb'] * 4:
                raise RuntimeError(f'the retained verb program changed: {methods}')
            captures.append(dict(id=docs[0]['id'], text=docs[0]['sentences'][0], tokens=tokens,
                read_order=[source_tokens[position] for position in read_positions],
                steps=steps, closed_meaning_norm=float(state[1][9][0, sid, start:end].norm())))
        return result
    model._sentence_observation = observed
    try:
        present_documents(model, docs)
    finally:
        model._sentence_observation = observe
    if len(captures) != 1:
        raise RuntimeError(f'expected one admitted clause trace, found {len(captures)}')
    return dict(checkpoint='accepted R2 confounded gate-on final checkpoint', captures=captures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('singletons', 'trace'))
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    restore = Models.BaseModel.load_weights
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = make_model(args.out, patch)
        try:
            restore_warmup(model, args.checkpoint, restore)
            model._concept_owner()._online_learning_frozen = True
            with torch.no_grad():
                report = singletons(model) if args.mode == 'singletons' else trace(model)
            save(args.out/'results.json', report)
        finally:
            model.End()


if __name__ == '__main__':
    main()
