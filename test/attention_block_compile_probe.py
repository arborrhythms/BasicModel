"""Probe the native sentence body's fullgraph boundary without changing it.

The eager compiler backend tests Dynamo capture only. It is neither an
accelerator benchmark nor a claim of accelerator code generation.
"""
import argparse
from collections import Counter
import linecache
import logging
import json
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from attention_thinking_certificate import lesson, restore


class ProbeComplete(Exception):
    pass


def inventory_kind(file, function, reason):
    """The object at the break, using the five categories of §53.3."""
    if file in ('Occurrence.py', 'Extension.py', 'ReferenceClose.py', 'ClauseRow.py', 'SentenceStrata.py',
                'PartClosing.py', 'MemoryLink.py', 'Meaning.py', 'ClauseJournal.py', 'ChunkJournal.py', 'Comprehension.py',
                'Understanding.py', 'ReadingIndex.py', 'ReferenceContext.py', 'HeldConcepts.py',
                'Interpret.py', 'IdeaStack.py', 'IdentityRows.py', 'FormCentroids.py'):
        return 'M'
    if file in ('CaseSelection.py',) or 'menu' in function or 'propose_actions' in function:
        return 'U'
    if file in ('IntuitionField.py', 'LookForecast.py', 'WhereRegistry.py', 'MemoryActivity.py',
                'FamiliarTier.py', 'Expectation.py', 'FieldReverse.py', 'Memory.py', 'Layers.py', 'Spaces.py', 'BasicPercepts.py', 'FeatureTable.py',
                'MereologicalCodes.py', 'PerceptField.py', 'ConceptTrust.py', 'SentenceField.py'):
        return 'T'
    if file in ('Language.py', 'WordIdentity.py', 'PrependedSymbols.py') or 'string' in reason.lower():
        return 'X'
    return 'C'


def resumed_report(function, args, kwargs):
    """Run the resumable compiler on a disposable model. Retain eager fallbacks too."""
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters, graph_break_reasons
    warnings = []
    class Fallbacks(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if "WON'T CONVERT" in message or 'recompile_limit' in message:
                warnings.append(message.splitlines()[0])
    log = logging.getLogger('torch._dynamo.convert_frame')
    handler = Fallbacks(logging.WARNING)
    log.addHandler(handler)
    source = Path(function.__code__.co_filename).resolve().parent
    report = dict(mode='explain', completed=False)
    try:
        with dynamo.config.patch(suppress_errors=True):
            explanation = dynamo.explain(function)(*args, **kwargs)
        report.update(completed=True, graph_count=explanation.graph_count,
                      graph_break_count=explanation.graph_break_count)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    finally:
        log.removeHandler(handler)
    # explain resets Dynamo on a successful return. Its result retains the
    # per-graph reasons after that reset; failures leave the live log intact.
    reasons = explanation.break_reasons if report['completed'] else graph_break_reasons
    sites = Counter()
    for event in reasons:
        if not event.graph_break:
            continue
        stack = [frame for frame in event.user_stack if Path(frame.filename).resolve().parent == source]
        frame = (stack or event.user_stack)[-1] if stack or event.user_stack else None
        file, line, function_name = ('<compiler>', 0, '') if frame is None else (
            Path(frame.filename).name, frame.lineno, frame.name)
        reason = event.reason.splitlines()[0]
        expression = '' if frame is None else linecache.getline(frame.filename, line).strip()
        sites[file, line, function_name, expression, reason, inventory_kind(file, function_name, reason)] += 1
    report['sites'] = [dict(file=f, line=n, function=name, expression=x, reason=r, kind=k, count=v)
                       for (f,n,name,x,r,k),v in sites.most_common()]
    report['fallbacks'] = dict(Counter(warnings))
    report['compiler_breaks'] = dict(counters['graph_break'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--explain', action='store_true',
                        help='resume after graph breaks and report location, reason and multiplicity')
    parser.add_argument('--resume-past', action='append', default=[], metavar='MODULE[.CLASS].FUNCTION',
                        help='in a follow-up explain run, execute an already diagnosed compiler failure eagerly')
    parser.add_argument('--follow-conditions', action='store_true',
                        help='on a follow-up, trace the branch actually taken after diagnosing an opaque torch.cond')
    args = parser.parse_args()
    if (args.resume_past or args.follow_conditions) and not args.explain:
        parser.error('diagnostic continuations require --explain')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    from Models import BasicModel
    with MonkeyPatch.context() as patch:
        use_eager_reading(patch)
        if args.follow_conditions:
            def native_branch(pred, true_fn, false_fn, operands):
                return true_fn(*operands) if bool(pred) else false_fn(*operands)
            patch.setattr(torch, 'cond', native_branch)
        eager_calls = Counter()
        def fallback(name, method):
            @torch.compiler.disable
            def call(*args, **kwargs):
                eager_calls[name] += 1
                return method(*args, **kwargs)
            return call
        for name in args.resume_past:
            module, *parents, method = name.split('.')
            owner = importlib.import_module(module)
            for parent in parents:
                owner = getattr(owner, parent)
            patch.setattr(owner, method, fallback(name, getattr(owner, method)))
        original = BasicModel._run_sentence_word_bricks
        compiled = torch.compile(original,
                                 backend='eager', fullgraph=True, dynamic=True)
        report = dict(backend='eager', fullgraph=True, dynamic=True,
                      fixture='native first lesson sentence', seed_selected=False)
        def attempt(*values, **kwargs):
            if args.explain:
                report.update(fullgraph=False, dynamic='automatic')
                report.update(resumed_report(original, values, kwargs))
                report.update(resumed_past=dict(eager_calls), followed_conditions=args.follow_conditions)
                raise ProbeComplete
            try:
                compiled(*values, **kwargs)
            except (torch._dynamo.exc.Unsupported, torch._dynamo.exc.UserError) as error:
                report.update(captured=False, error_type=type(error).__name__, error=str(error))
            else:
                report.update(captured=True)
            raise ProbeComplete
        patch.setattr(BasicModel, '_run_sentence_word_bricks', attempt)
        metadata = json.loads((args.replay/'initial.json').read_text())
        assert os.environ.get('PYTHONHASHSEED') == metadata['pythonhashseed']
        restore(torch.load(args.replay/'initial-rng.pt', weights_only=False))
        observer = SimpleNamespace(reference_seconds=0., records=[], calls=[], actions=[], costs=[])
        try:
            lesson(observer)
        except ProbeComplete:
            pass
        assert 'captured' in report or 'completed' in report, 'the native sentence body was not reached'
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('compiler_breaks', 'sites', 'fallbacks')}, indent=2))


if __name__ == '__main__':
    main()
