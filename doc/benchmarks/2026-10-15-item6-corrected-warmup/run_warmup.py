"""Repeat the accepted stages 1–4 budgets under the corrected production objective.

Reuse the existing training loop. Suppress only archive hashes; observe native
singleton programs during their original presentations, without extra exposure.
"""
import argparse
from pathlib import Path
import warnings

import pytest
import torch
from torch.nn import functional as F

import attention_curriculum_gates as curriculum
import identity_experiment as experiment
from attention_curriculum_gates import PERCEPTS, INTERRUPTION, NOVEL
from identity_corpus import KINDS, PROPERTIES, VERBS


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
    return dict(checkpoint='corrected-objective stages 1–4 warm-up', meaning_slice=[chart.meaning_start, chart.content_width],
        singleton_selection='native observation, order 0, one reference to a stage-1/2 word concept',
        atom_meaning_norms={form: float(value[chart.meaning_start:chart.content_width].norm())
                            for form, value in atoms.items()},
        singleton_rows=rows, missing_singleton_words=sorted(set(forms)-{r['word'] for r in rows}),
        warmup_columns=columns, atom_space='native dictionary event coordinates',
        column_match_threshold=.9)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    current = {}
    traces = []
    models = []
    atoms = set((*PERCEPTS, *INTERRUPTION, *NOVEL, *KINDS, *PROPERTIES, *VERBS))
    native_make, native_present, native_save = experiment.make_model, curriculum.present, experiment.save

    def present(model, stage, **kwargs):
        current.clear()
        current.update(fields=stage['fields'], training=bool(kwargs.get('train', False)))
        return native_present(model, stage, **kwargs)

    def make(folder, patch):
        model = native_make(folder, patch)
        models.append(model)
        observe = model._sentence_observation
        def observed(state, sid, active, **kwargs):
            result = observe(state, sid, active, **kwargs)
            if kwargs.get('admit') and all(word in atoms for word in current.get('fields', ())):
                chart = model._concept_owner().components._ready()
                start, end = chart.meaning_start, chart.content_width
                sources = model._derivation_program(int(sid))[0]
                for b, word in enumerate(current['fields']):
                    program = result['entries'][b]
                    if program is None:
                        raise RuntimeError(f'{word}: admitted singleton has no program')
                    positions = sources[b].tolist()
                    positions = [int(p) for p in positions if p >= 0]
                    steps = []
                    for index, (kind, operation, leaf) in enumerate(program.actions.tolist()):
                        rules = (model.languageSpace._compose_binary_rules if kind == 1 else
                                 model.languageSpace._compose_unary_rules if kind == 2 else ())
                        frame = program.operation_values[index, :, start:end].detach()
                        value = model._tensor_pushed_ideas[b, positions[leaf], start:end].detach() if kind == 0 else None
                        steps.append(dict(step=index, kind=kind, operation=operation,
                            method=rules[operation].method_name if rules else None,
                            leaf_position=positions[leaf] if kind == 0 else None,
                            leaf_meaning_norm=float(value.norm()) if value is not None else None,
                            meaning_norms=frame.norm(dim=-1).tolist() if kind else None))
                    traces.append(dict(word=word, training=current['training'], steps=steps,
                        closed_meaning_norm=float(state[1][9][b, sid, start:end].detach().norm())))
                native_save(args.out/'singleton-presentations.json', traces)
            return result
        model._sentence_observation = observed
        return model

    def save(path, value):
        if path.name not in ('state-hashes.json', 'checkpoint.json'):
            native_save(path, value)

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        patch.setattr(experiment, 'make_model', make)
        patch.setattr(curriculum, 'present', present)
        patch.setattr(experiment, 'save', save)
        patch.setattr(experiment, 'model_hashes', lambda model: None)
        patch.setattr(experiment, 'digest', lambda path: None)
        experiment.warmup(args.out, 64, 8)
        model = models[0]
        with torch.no_grad():
            report = singletons(model)
        if len(report['singleton_rows']) != 24 or len({row['word'] for row in report['singleton_rows']}) != 24:
            raise RuntimeError('the singleton gate requires one stored row for each of the 24 words')
        report['gate'] = dict(threshold=.9, required=20, total=24,
            carrying=sum(row['cosine'] is not None and row['cosine'] >= .9
                         for row in report['singleton_rows']))
        report['gate']['passed'] = (not report['missing_singleton_words'] and
            report['gate']['carrying'] >= report['gate']['required'])
        report.update(seed=20261010, order_steps=64, sentence_steps=8,
                      source_landing='f1dbd12a2', completed=True)
        native_save(args.out/'singletons.json', report)
        print(report['gate'], flush=True)


if __name__ == '__main__':
    main()
