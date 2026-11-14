"""Certify sentence-owned admission; record authorized differences from base.

Every individual attention kernel still matches the original oracle exactly.
Semantic before/after differences are reported separately from later, exact
host-work optimizations, which use --compare against the sentence-owned run.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from attention_thinking_certificate import Observer, frozen, gates, lesson, restore, stream
from attention_single_walk_certificate import packed


class SentenceOwnerObserver(Observer):
    def __init__(self, patch, output):
        import ModelAttention
        from Models import BasicModel
        super().__init__(patch, output, certify=True)
        self.audit = []
        self.expected = {}
        self.owners = set()
        stage = ModelAttention.stage_input
        def staged(model):
            before = len(self.calls)
            result = stage(model)
            if model._sentence_ends:
                assert len(self.calls) == before, 'sentence staging still walks the row'
                assert model._attention_words is None and model._attention_greedy is None
                assert model._word_expectation is None
            owner = getattr(model.symbolSpace, 'expectation', None)
            if owner is not None and id(owner) not in self.owners:
                self.owners.add(id(owner))
                expect = owner.expect
                def forecast(level, *args, **kwargs):
                    value = expect(level, *args, **kwargs)
                    if level == 'word':
                        sid = getattr(model, '_open_sentence_slot', None)
                        key = id(model), sid
                        assert key not in self.expected, 'word expectation rebuilt within a sentence'
                        self.expected[key] = value
                        self.audit.append(dict(event='expectation', sentence=sid,
                            clock=model.present(), targets=args[3].detach().tolist(),
                            active=kwargs['active'].detach().tolist()))
                    return value
                patch.setattr(owner, 'expect', forecast)
            return result
        patch.setattr(ModelAttention, 'stage_input', staged)
        path_cost = BasicModel._sentence_path_cost
        def cost(model, *args, **kwargs):
            sid = model._open_sentence_slot
            value = getattr(model, '_word_expectation', None)
            if value is not None:
                assert value is self.expected[id(model), sid], 'path changed the shared expectation'
                before = frozen(value.loss)
            result = path_cost(model, *args, **kwargs)
            if value is not None:
                assert frozen(value.loss) == before, 'path mutated the shared expectation'
            meters = getattr(model, '_attention_meters', ())
            assert all(0 <= meter.spent <= meter._limit for meter in meters)
            self.audit.append(dict(event='path', sentence=sid,
                                   meters=[dict(meter.counts) for meter in meters]))
            return result
        patch.setattr(BasicModel, '_sentence_path_cost', cost)

    def batch(self, model, **kwargs):
        index = len(self.records)
        super().batch(model, **kwargs)
        (self.output/f'ownership-{index:03}.json').write_text(json.dumps(self.audit,indent=2)+'\n')
        self.audit.clear()
        self.expected.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('lesson', 'gates', 'packed'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--report-base', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    with MonkeyPatch.context() as patch:
        use_eager_reading(patch)
        observer = SentenceOwnerObserver(patch, args.output)
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
                      oracle_seconds=observer.reference_seconds, records=observer.records)
        torch.save(stream(), args.output/'final-rng.pt')
        for reference, required in ((args.compare, True), (args.report_base, False)):
            if reference is None:
                continue
            prior = sorted(reference.glob('batch-*.json'))
            assert len(prior) == len(observer.records)
            differences = []
            for old in prior:
                before = json.loads(old.read_text())
                after = json.loads((args.output/old.name).read_text())
                changed = [key for key in before.keys() | after.keys() if before.get(key) != after.get(key)]
                if changed:
                    differences.append(dict(batch=old.name, fields=changed))
            if required:
                assert not differences, differences
                report['matches_reference_exactly'] = True
            else:
                report['base_differences'] = differences
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
