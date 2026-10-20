"""Read-only diagnostics for the existing c/e certificate."""
import json
import random
from pathlib import Path
import os
import shutil
import pytest


@pytest.fixture(autouse=True)
def observe_closing(monkeypatch, request):
    import torch
    import numpy as np
    from Models import BasicModel
    from ThoughtReferences import bindings, open_slots
    if 'forced_c_e_closings' in request.node.nodeid:
        # Retain the failing initialization if this diagnostic finds another
        # resource edge. This is a replay aid, never a passing-seed selection.
        initial = Path(os.environ['CLOSING_TRACE']).with_suffix('.initial-rng.pt')
        if initial.exists():
            raise RuntimeError('refusing to overwrite a closing initialization')
        replay = os.environ.get('CLOSING_REPLAY')
        if replay:
            rng = torch.load(replay, weights_only=False)
            torch.set_rng_state(rng['torch'])
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
        torch.save(dict(torch=torch.get_rng_state(), python=random.getstate(),
                        numpy=np.random.get_state()), initial)
    original = BasicModel._run_selected_thought_once
    def run(model, meaning, **kwargs):
        if 'forced_c_e_closings' in request.node.nodeid:
            from Spaces import _concept_alloc_of
            owner = model._concept_owner()
            alloc, store = _concept_alloc_of(owner), model.symbolSpace.ltm_store
            retained = []
            for index in range(len(store)):
                value = store.raw_meaning_of(index)
                retained.append(dict(index=index, occurrence=store.occurrence_of(index),
                    refs=None if value is None else value.role_refs,
                    open=() if value is None else open_slots(value),
                    evidence=(float(store.c_plus[index]), float(store.c_minus[index]))))
            snapshot = dict(stream=kwargs['row'], query_refs=meaning.role_refs,
                words={str(k): list(v) for k, v in alloc.word_forms.items()},
                singletons=sorted(alloc.singletons),
                types={str(k): sorted(v) for k, v in alloc.symbol_types.items()},
                parts={int(i): {key: value.tolist() for key, value in store.parts.row(i).items()}
                       for i in store.parts.indices().tolist()},
                occurrences=retained)
            with Path(os.environ['CLOSING_TRACE']).with_suffix('.store.jsonl').open('a') as stream:
                stream.write(json.dumps(snapshot, default=str) + '\n')
        result = original(model, meaning, **kwargs)
        value = dict(test=request.node.nodeid, row=kwargs['row'],
            refs=meaning.role_refs, bindings=bindings(meaning),
            after_refs=result.meaning.role_refs, after_bindings=bindings(result.meaning),
            work=result.work.spent, counts=result.work._counts,
            records=[dict(kind=record.kind, operation=record.operation,
                refs=None if record.meaning is None else record.meaning.role_refs,
                open=() if record.meaning is None else open_slots(record.meaning),
                incomplete=() if record.result is None else record.result.incomplete)
                for record in result.records])
        value['reads'] = [dict(operation=record.operation,
            frames=[dict(occurrence=frame.get('occurrence'), refs=frame['meaning'].role_refs,
                         evidence=frame.get('evidence'), kind=frame.get('kind'),
                         bindings=bindings(frame['meaning']))
                    for frame in record.result.evidence.get('frames', ())])
            for record in result.records if record.result is not None]
        questions = getattr(model, '_sentence_answer_questions', ())
        if kwargs['row'] < len(questions):
            answer = model.inputSpace.data.what(questions[kwargs['row']])
            if answer.available:
                from BindingAnswers import cost
                value['binding_cost'] = float(cost(model, result.meaning, answer.what, stream=kwargs['row']))
        with Path(os.environ['CLOSING_TRACE']).open('a') as stream:
            stream.write(json.dumps(value, default=str)+'\n')
        return result
    monkeypatch.setattr(BasicModel, '_run_selected_thought_once', run)
    yield
    folder = request.node.funcargs.get('tmp_path')
    destination = os.environ.get('SCOPED_REPORTS')
    if folder is not None and destination:
        target = Path(destination)/request.node.name
        target.mkdir(parents=True, exist_ok=True)
        for source in folder.rglob('*'):
            if source.is_file() and source.suffix in ('.json', '.jsonl'):
                output = target/source.relative_to(folder)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, output)
