"""Checkpoint-only stage 4e case and chooser probe; never continues the run."""
import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch

from constituent_corpus import DEST, load
from identity_experiment import make_model, restore_warmup, save


def probe(args):
    from Models import BaseModel
    from What import What
    from GrammarLessons import constituent_examples, compose_loss, ComposeLessonStates, student_target_probability
    from Layers import Error
    from ObjectiveOwnership import backward_owned, registry_costs
    restore = BaseModel.load_weights
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(20261010); np.random.seed(20261010); torch.manual_seed(20261010)
    texts, labels = load('train')
    annotations = json.loads((DEST/'train.lessons.json').read_text())
    chosen = [next(i for i, row in enumerate(texts) if labels[row['id']]['family'] == family)
              for family in ('np', 'intransitive', 'transitive', 'preposition')]
    lessons = [annotations[i] for i in chosen]
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = make_model(args.out, patch)
        try:
            drift = restore_warmup(model, args.checkpoint, restore)
            programs = []
            if args.states:
                programs = torch.load(args.states, weights_only=False)
            else:
                from contextlib import contextmanager
                native_run = model._sentence_run
                @contextmanager
                def observed_run(**kwargs):
                    with native_run(**kwargs):
                        model.languageSpace._compose_lesson_states = ComposeLessonStates()
                        yield
                model._sentence_run = observed_run
                original = model._sentence_observation
                def capture(state, sid, active, **kwargs):
                    result = original(state, sid, active, **kwargs)
                    if kwargs.get('admit'):
                        positions = model._derivation_program(int(sid))[0]
                        for b, program in enumerate(result['entries']):
                            order = positions[b][positions[b] >= 0].argsort()
                            view = SimpleNamespace(**{name: getattr(program, name).detach().index_select(0, order)
                                for name in ('leaves', 'concept_ids', 'word_rows')})
                            width = model.inputSpace._word_active_mask.shape[1]
                            active = model.inputSpace._word_active_mask[b]
                            actual = positions[b][positions[b] >= 0]
                            closing_width = 2*int(model.conceptualSpace.stm.capacity)
                            hi, last = int(actual.max()), int(active.nonzero()[-1])
                            base = 3*width if hi == last else 3*width+(hi+1)*closing_width
                            view.student_states = model.languageSpace._compose_lesson_states.sentence(
                                b, actual, width=width, closing_base=base, closing_width=closing_width,
                                attempted=model._reconstruction_stack()._choice_attempted)
                            view.case_bank = model.languageSpace._reference_bank.cases.detached()
                            programs.append(view)
                    return result
                model._sentence_observation = capture
                model._concept_owner()._online_learning_frozen = True
                data = model.inputSpace.data
                data.grammar_lessons = {}
                data.source_addresses = {'validation': [dict(document=row['id'], sentence=0,
                    document_key=row['id'], sentence_index=0, split='validation') for row in texts]}
                with torch.no_grad():
                    inputs = model.inputSpace.prepPackedInput([[texts[i]['text']] for i in chosen])
                    model.runBatch(train=False, optimizer=None, batchNum=0, batchSize=4, split='validation',
                        batch_override=(inputs, torch.empty(4, 0)), source_rows=[[i] for i in chosen],
                        questions=tuple(What.present(b, split='validation') for b in range(4)))
                model._sentence_observation = original
                # The immutable grammar capability contains MappingProxy data;
                # diagnostics report observations in JSON instead of pickling it.
            assert len(programs) == len(lessons)
            if any(not hasattr(program, 'student_states') for program in programs):
                raise ValueError('retired teacher-state checkpoints require their original code revision')
            layer = model.languageSpace._tree_layer(2)
            chooser = layer.chooser
            before = {name: p.detach().clone() for name, p in chooser.named_parameters()}
            optimizer = model.getOptimizer(lr=.003)
            owned = {id(p) for group in optimizer.param_groups for p in group['params']}
            groups = model.objective_parameter_groups(optimizer)
            lesson_owned = {id(p) for p in groups['compose_lesson']}
            raw_hidden, preactivation, raw_scores, routes = [], [], [], []
            def hidden(_module, _inputs, value):
                raw_hidden.append(value.detach().flatten())
            def route(_module, _inputs, value):
                routes.append(value[2])
            def nonlinear(_module, inputs):
                preactivation.append(inputs[0].detach().flatten())
            original_score = chooser._score
            def score(*values, **kwargs):
                result = original_score(*values, **kwargs)
                if result.numel():
                    raw_scores.append(dict(shape=list(result.shape),
                        gap=float((result.detach().amax(-1)-result.detach().amin(-1)).max())))
                return result
            chooser._score = score
            hook = chooser.mlp[0].register_forward_hook(hidden)
            nonlinear_hook = chooser.mlp[1].register_forward_pre_hook(nonlinear)
            route_hook = layer.register_forward_hook(route)
            registry = Error()
            loss = compose_loss(model.languageSpace, programs, lessons, errors=registry)
            hook.remove(); route_hook.remove(); nonlinear_hook.remove()
            chooser._score = original_score
            optimizer.zero_grad()
            direct = torch.autograd.grad(loss, tuple(chooser.parameters()), retain_graph=True, allow_unused=True)
            backward_owned(registry_costs(registry), groups)
            stats = []
            for (name, p), gradient in zip(chooser.named_parameters(), direct):
                stats.append(dict(name=name, optimizer_owned=id(p) in owned,
                    lesson_owned=id(p) in lesson_owned,
                    direct_gradient_norm=None if gradient is None else float(gradient.norm()),
                    gradient_norm=None if p.grad is None else float(p.grad.norm()),
                    parameter_norm=float(p.detach().norm())))
            optimizer.step()
            for row in stats:
                delta = dict(chooser.named_parameters())[row['name']].detach()-before[row['name']]
                row.update(change_norm=float(delta.norm()), change_max=float(delta.abs().max()))
            after_loss = compose_loss(model.languageSpace, programs, lessons)
            cases, states = [], []
            for lesson, program in zip(lessons, programs):
                bank = program.case_bank
                examples = constituent_examples(model.languageSpace, program, lesson['tree'])
                cases.append(dict(text=lesson['text'], ids=bank.ids.tolist(),
                    cases_nonzero=int(bank.cases.count_nonzero()), observed_nonzero=int(bank.observed.count_nonzero()),
                    examples=len(examples)))
                for ex in examples:
                    route = routes[len(states)]
                    logits = route['logits'].detach()[ex['row']]
                    finite = logits[torch.isfinite(logits)]
                    target = ex['target']
                    states.append(dict(text=lesson['text'], target=target, depth=ex['depth'],
                        logits=[None if not torch.isfinite(v) else float(v) for v in logits],
                        logit_gap=float(finite.max()-finite.min()),
                        target_probability=float(student_target_probability(layer, ex, route['probabilities'][ex['row']]).detach())))
            h = torch.cat(raw_hidden)
            z = torch.cat(preactivation)
            owner = model._concept_owner()
            store = owner._concept_allocator.layer(0)
            report = dict(checkpoint=str(args.checkpoint), student_states=None if args.states is None else str(args.states),
                loader_drift=drift, case_bank=cases, states=states, parameters=stats,
                terms=registry.breakdown(), raw_score_gaps=raw_scores,
                loss_before=float(loss.detach()), loss_after_one_step=float(after_loss.detach()),
                first_linear=dict(min=float(h.min()), max=float(h.max()),
                    fraction_below_minus5=float((h < -5).float().mean()),
                    fraction_above5=float((h > 5).float().mean())),
                nonlinearity=dict(kind=type(chooser.mlp[1]).__name__, min=float(z.min()), max=float(z.max()),
                    fraction_below_minus5=float((z < -5).float().mean())),
                sigma=dict(edges=len(store._rows), positive_edges=0 if store.values is None else int((store.values > 0).sum())))
            save(args.out/'diagnostic.json', report)
            print(json.dumps({key: report[key] for key in ('loss_before', 'loss_after_one_step', 'first_linear', 'sigma')}), flush=True)
        finally:
            model.End()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('output/stage4e-20261017-r3/trained.ckpt'))
    parser.add_argument('--states', type=Path)
    probe(parser.parse_args())
