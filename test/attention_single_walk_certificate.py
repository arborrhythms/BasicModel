"""Compare staging and sentence attention before changing their ownership.

Uses the section-7 recorded stream and observer. Production state, capacities,
assertions and choices are unchanged. Diagnostic expectation calls run on
attribute overlays and a private copy of the owner's nonpersistent bank.
"""
import argparse
from copy import copy
from dataclasses import fields
import inspect
import json
import os
from pathlib import Path
import time

import torch

from attention_thinking_certificate import Observer, frozen, gates, lesson, restore, stream


def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        items = [clone(item) for item in value]
        return type(value)(*items) if hasattr(value, '_fields') else type(value)(items)
    return value


class Overlay:
    def __init__(self, source, **values):
        self.__dict__.update(_source=source, **values)

    def __getattr__(self, name):
        return getattr(self._source, name)


def walk_payload(reading):
    # The Git-loaded oracle's NamedTuple class is deliberately not installed
    # as a production module. Save plain containers for portable evidence.
    return clone(dict(reading._asdict(), table=dict(reading.table._asdict())))


def expectation_payload(model):
    inputs = getattr(model, '_word_expectation_input', None)
    expected = getattr(model, '_word_expectation', None)
    if inputs is None or expected is None:
        return None
    owner = model.symbolSpace.expectation
    return clone(dict(inputs=dict(zip(('codes', 'bank', 'valid', 'targets'), inputs)),
        expected={field.name: getattr(expected, field.name) for field in fields(expected)},
        symbol_bank={name: getattr(owner, name) for name in
                     ('_bank_keys', '_bank_ids', '_bank_addresses', '_bank_orders')},
        mask=model._word_expectation_mask, column=model.conceptualSpace._word_column,
        leaf_mask=model.inputSpace._ar_grammar_leaf_mask, handoff_poles=model._attention_poles))


def scoped(payload, mask):
    if payload is None:
        return None
    return dict(inputs={name: value[mask] if name in ('codes', 'targets') else value
                        for name, value in payload['inputs'].items()},
        expected={name: value[mask] for name, value in payload['expected'].items()},
        symbol_bank=payload['symbol_bank'],
        **{name: payload[name][mask] for name in ('mask', 'column', 'leaf_mask', 'handoff_poles')})


def differences(before, after, prefix=''):
    result = {}
    if isinstance(before, dict) and isinstance(after, dict):
        assert before.keys() == after.keys()
        for name in before:
            result.update(differences(before[name], after[name], prefix+name+'.'))
    elif torch.is_tensor(before) and torch.is_tensor(after):
        key = prefix.rstrip('.')
        if before.shape != after.shape or before.dtype != after.dtype:
            result[key] = dict(before=frozen(before), after=frozen(after))
        elif not torch.equal(before, after):
            different = before != after
            positions = different.nonzero()[:8]
            at = tuple(positions.T)
            result[key] = dict(elements=int(different.sum()), positions=positions.tolist(),
                before=before[at].tolist(), after=after[at].tolist(),
                max_abs=float((before.double()-after.double()).abs().max()))
    elif before != after:
        result[prefix.rstrip('.')] = dict(before=before, after=after)
    return result


def probe(model, reading, *, clock=None, codes=None, contexts=None):
    """Execute the real handoff/expectation on isolated assignment owners."""
    import ModelAttention
    owner = model.symbolSpace.expectation
    private = copy(owner)
    private._buffers = owner._buffers.copy()
    if clock is not None and owner._ltm_store is not None:
        object.__setattr__(private, '_ltm_store', Overlay(owner._ltm_store, _model_present=lambda: clock))
    if contexts is not None:
        private._read_context = lambda row: contexts[row]
    isp = Overlay(model.inputSpace)
    if codes is not None:
        isp._ar_word_object_atoms = codes
    view = Overlay(model, inputSpace=isp, conceptualSpace=Overlay(model.conceptualSpace),
                   symbolSpace=Overlay(model.symbolSpace, expectation=private))
    ModelAttention.handoff(view, reading)
    ModelAttention.stage_expectation(view)
    return expectation_payload(view)


class SoleWalkObserver(Observer):
    def __init__(self, patch, output, *, certify):
        import Attention, ModelAttention
        from test_attention_thinking_steps import original_attention
        super().__init__(patch, output, certify=certify)
        self.contexts, self.comparisons, self.summaries = {}, [], []
        self.latest_open = None
        self.in_probe = False
        base = original_attention()
        opening = Attention.open_narrowing
        def opened(*args, **kwargs):
            self.latest_open = opening(*args, **kwargs)
            return self.latest_open
        patch.setattr(Attention, 'open_narrowing', opened)
        patch.setattr(ModelAttention, 'open_narrowing', opened)

        staging = ModelAttention.stage_input
        def stage_input(model):
            result = staging(model)
            reading = getattr(model, '_attention_greedy', None)
            if reading is not None:
                self.contexts[id(model)] = dict(reading=clone(reading),
                    arguments=self.latest_open._certificate_arguments,
                    sentence_ids=model.inputSpace._packed_sentence_ids.detach().clone(),
                    forms=clone(model._attention_forms[0]),
                    initial_meters=[meter.spent for meter in model._attention_meters],
                    staging_rounds=self.latest_open.steps_taken)
            return result
        patch.setattr(ModelAttention, 'stage_input', stage_input)

        expectation = ModelAttention.stage_expectation
        def stage_expectation(model):
            result = expectation(model)
            if self.in_probe or isinstance(model, Overlay) or id(model) not in self.contexts:
                return result
            saved = self.contexts[id(model)]
            if 'expectation' in saved:
                return result
            started = time.perf_counter()
            saved['expectation'] = expectation_payload(model)
            saved['clock'] = model.present()
            owner = model.symbolSpace.expectation
            saved['contexts'] = clone([owner._read_context(row) for row in range(len(saved['reading'].accepted))])
            if saved['expectation'] is not None:
                check = self.probe(model, saved['reading'])
                assert not differences(saved['expectation'], check), 'isolated probe changed staging expectation'
            chooser, keys, spans, known, kwargs = saved['arguments']
            saved['cold_sentences'] = []
            cold_readings = []
            for sid in range(model.inputSpace._packed_sentence_slot_mask.shape[1]):
                here = saved['sentence_ids'] == sid
                if not bool((here & model.inputSpace._word_active_mask).any()):
                    continue
                local_spans = torch.where(here[:, :spans.shape[1], None], spans, 0)
                local = base.narrow_words(chooser, keys, local_spans, known, **kwargs)
                local = ModelAttention._word_carrier(local, here.shape[1])
                cold_readings.append(walk_payload(local))
                value = self.probe(model, local)
                saved['cold_sentences'].append(dict(sentence=sid,
                    reading=self.reading_difference(saved['reading'], local, here),
                    expectation=differences(scoped(saved['expectation'], here), scoped(value, here)),
                    staging_spent=saved['reading'].table.spent.tolist(), local_spent=local.table.spent.tolist(),
                    staging_accepted=saved['reading'].accepted[here].tolist(), local_accepted=local.accepted[here].tolist(),
                    staging_descended=saved['reading'].descended[here].tolist(), local_descended=local.descended[here].tolist(),
                    staging_poles=saved['reading'].poles[here].tolist(), local_poles=local.poles[here].tolist()))
                if sid:
                    # Isolate the causal prefix from clock/history changes:
                    # restore only earlier words from the same staging read.
                    positions = torch.arange(here.shape[1], device=here.device)[None]
                    first = torch.where(here, positions, here.shape[1]).amin(-1)
                    earlier = positions < first[:, None]
                    prefixed = local._replace(**{name: torch.where(
                        earlier[..., None] if getattr(local, name).ndim == 3 else earlier,
                        getattr(saved['reading'], name), getattr(local, name))
                        for name in ('values', 'accepted', 'descended', 'poles', 'pole_changes')})
                    value = self.probe(model, prefixed)
                    saved['cold_sentences'][-1]['expectation_with_staged_prefix'] = differences(
                        scoped(saved['expectation'], here), scoped(value, here))
            torch.save(dict(staging=walk_payload(saved['reading']), sentences=cold_readings,
                            initial_meters=saved['initial_meters']),
                       output/f'walks-{len(self.records):03}.pt')
            self.reference_seconds += time.perf_counter()-started
            return result
        patch.setattr(ModelAttention, 'stage_expectation', stage_expectation)

        handoff = ModelAttention.handoff
        def handed(model, reading):
            result = handoff(model, reading)
            if self.in_probe or isinstance(model, Overlay) or id(model) not in self.contexts:
                return result
            caller = inspect.currentframe().f_back
            at_start = caller.f_code.co_name == 'start_compose'
            del caller
            if not at_start or model._sentence_trial != 'exploit':
                return result
            started = time.perf_counter()
            saved = self.contexts[id(model)]
            sid = int(model._open_sentence_slot)
            here = saved['sentence_ids'] == sid
            current_staging = self.probe(model, saved['reading'])
            current_local = self.probe(model, reading)
            at_old_clock = self.probe(model, reading, clock=saved['clock'])
            frozen_inputs = self.probe(model, reading, clock=saved['clock'],
                codes=None if saved['expectation'] is None else saved['expectation']['inputs']['codes'],
                contexts=saved['contexts'])
            comparison = dict(sentence=sid, clock_before=saved['clock'], clock_at_word=model.present(),
                forms=saved['forms'], sentence_ids=saved['sentence_ids'].tolist(),
                staging_rounds=saved['staging_rounds'],
                staging_spent=saved['reading'].table.spent.tolist(), local_spent=reading.table.spent.tolist(),
                meters_at_word=[meter.spent for meter in model._attention_meters],
                reading=self.reading_difference(saved['reading'], reading, here),
                expectation_same_time=differences(scoped(current_staging, here), scoped(current_local, here)),
                expectation_moved=differences(scoped(saved['expectation'], here), scoped(current_local, here)),
                expectation_old_clock=differences(scoped(saved['expectation'], here), scoped(at_old_clock, here)),
                expectation_frozen_inputs=differences(scoped(saved['expectation'], here), scoped(frozen_inputs, here)))
            index = len(self.records), len(self.comparisons)
            torch.save(dict(staging_reading=walk_payload(saved['reading']), sentence_reading=walk_payload(reading),
                            staging=saved['expectation'], current_staging=current_staging,
                            current_local=current_local, old_clock=at_old_clock, frozen_inputs=frozen_inputs),
                       output/f'expectations-{index[0]:03}-{index[1]:03}.pt')
            self.comparisons.append(comparison)
            (output/'comparisons-progress.json').write_text(json.dumps(self.comparisons, indent=2)+'\n')
            self.reference_seconds += time.perf_counter()-started
            return result
        patch.setattr(ModelAttention, 'handoff', handed)

    def probe(self, *args, **kwargs):
        initial = torch.get_rng_state()
        self.in_probe = True
        try:
            value = probe(*args, **kwargs)
        finally:
            self.in_probe = False
        assert torch.equal(torch.get_rng_state(), initial), 'diagnostic consumed a random draw'
        return value

    @staticmethod
    def reading_difference(before, after, mask):
        a, b = ({name: getattr(value, name)[mask] for name in ('accepted', 'descended', 'poles')}
                for value in (before, after))
        a['spent'], b['spent'] = before.table.spent, after.table.spent
        return differences(a, b)

    def batch(self, model, **kwargs):
        index = len(self.records)
        super().batch(model, **kwargs)
        saved = self.contexts.get(id(model))
        summary = dict(batch=index, label=kwargs['label'], comparisons=self.comparisons,
                       cold_sentences=[] if saved is None else saved['cold_sentences'])
        (self.output/f'walk-comparison-{index:03}.json').write_text(json.dumps(summary, indent=2)+'\n')
        self.summaries.append(summary)
        self.comparisons = []
        self.contexts.clear()


def packed(observer):
    """The complete first lesson document, packed at unchanged XML capacity."""
    import Language
    from Models import BaseModel
    from data import TheData
    from corpus_lessons import stage_documents
    from Corpus import admit_training_forms
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                      for split in ('train', 'validation', 'test')})
    Language.TheGrammar._configured = False
    model = BaseModel.from_config('data/MM_corpus_lessons.xml', data=TheData)[0]
    model._tensor_peer_while_eager = True
    stage = json.loads(Path('data/corpus-lessons.json').read_text())['stages'][0]
    texts, targets = stage_documents(TheData, stage)
    admit_training_forms(model, texts, targets)
    rows = [i for i, address in enumerate(TheData.source_addresses['train']) if address['document'] == 0]
    model.begin_presentation()
    inputs = model.inputSpace.prepPackedInput([[texts[i] for i in rows]])
    observer.calls, observer.actions, observer.costs = [], [], []
    started, reference = time.perf_counter(), observer.reference_seconds
    error = None
    try:
        model.runBatch(train=False, batchSize=1, split='train', source_rows=[rows],
                       batch_override=(inputs, torch.zeros(1, 1)))
    except Exception:
        import traceback
        error = traceback.format_exc()
        (observer.output/'packed-error.txt').write_text(error)
    finally:
        observer.batch(model, seconds=time.perf_counter()-started-(observer.reference_seconds-reference),
                       label=dict(rows=rows, texts=[texts[i] for i in rows], error=error))
        model.End()
    assert error is None, 'packed fixture failed; see packed-error.txt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('lesson', 'gates', 'packed'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    import Attention
    with MonkeyPatch.context() as patch:
        use_eager_reading(patch)
        # The identical driver also runs against the specified base checkout,
        # which predates the step API. Only the observer differs there.
        observer_type = SoleWalkObserver if hasattr(Attention, 'open_narrowing') else Observer
        observer = observer_type(patch, args.output, certify=True)
        metadata = json.loads((args.replay/'initial.json').read_text())
        assert os.environ.get('PYTHONHASHSEED') == metadata['pythonhashseed']
        restore(torch.load(args.replay/'initial-rng.pt', weights_only=False))
        torch.save(stream(), args.output/'initial-rng.pt')
        started = time.perf_counter()
        if args.kind == 'lesson':
            lesson(observer)
        elif args.kind == 'gates':
            gates(observer, patch)
        else:
            packed(observer)
        report = dict(kind=args.kind, seconds=time.perf_counter()-started,
                      diagnostic_seconds=observer.reference_seconds, records=observer.records)
        torch.save(stream(), args.output/'final-rng.pt')
        if args.compare:
            previous = json.loads((args.compare/'result.json').read_text())
            assert len(previous['records']) == len(observer.records)
            for index in range(len(observer.records)):
                name = f'batch-{index:03}.json'
                assert json.loads((args.compare/name).read_text()) == json.loads((args.output/name).read_text()), name
            report['matches_base_exactly'] = True
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        (args.output/'walk-comparisons.json').write_text(json.dumps(getattr(observer, 'summaries', []), indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
