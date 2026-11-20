"""A probe: what a sentence's looks call, counted on the lesson's
sentences. Run as a pytest file with -s; it asserts nothing."""
import json
from pathlib import Path

import torch


def _counted(monkeypatch):
    import Spaces
    import SentenceLooks
    counts = dict(looks=0, materialize_in_read=0, materialize_in_arrive=0, materialize_elsewhere=0,
                  host_reads_in_read=0)
    where = ['elsewhere']
    for cls in (Spaces.SubSpaceView, Spaces.SubSpace):
        original = cls.materialize
        def counted(self, *a, _original=original, **k):
            import sys
            caller = sys._getframe(1).f_code.co_name
            counts.setdefault('callers', {}).setdefault(caller, 0)
            counts['callers'][caller] += 1
            counts['materialize_' + ('in_' + where[-1] if where[-1] != 'elsewhere' else 'elsewhere')] += 1
            return _original(self, *a, **k)
        monkeypatch.setattr(cls, 'materialize', counted)
    host = {}
    for name in ('tolist', 'item', '__bool__', '__int__', '__float__', 'cpu'):
        original = getattr(torch.Tensor, name)
        def read(self, *a, _original=original, **k):
            if where[-1] == 'read':
                counts['host_reads_in_read'] += 1
                import sys
                frame = sys._getframe(1)
                key = frame.f_code.co_filename.rsplit('/', 1)[-1] + ':' + frame.f_code.co_name
                readers = counts.setdefault('readers', {})
                readers[key] = readers.get(key, 0) + 1
            return _original(self, *a, **k)
        host[name] = original
        monkeypatch.setattr(torch.Tensor, name, read)
    perceive = SentenceLooks.perceive
    def perceived(*a, **k):
        counts['looks'] += 1
        where.append('read')
        try:
            return perceive(*a, **k)
        finally:
            where.pop()
    monkeypatch.setattr(SentenceLooks, 'perceive', perceived)
    run = SentenceLooks.Looks.run
    def running(self, arrive, **k):
        def arriving(*a, **kk):
            where.append('arrive')
            try:
                return arrive(*a, **kk)
            finally:
                where.pop()
        return run(self, arriving, **k)
    monkeypatch.setattr(SentenceLooks.Looks, 'run', running)
    return counts


def test_count_a_looks_path(monkeypatch):
    from test_item17_order_zero import _read_lesson
    model, patch = _read_lesson(16)
    try:
        counts = _counted(monkeypatch)
        from test_item18_expect import read
        curriculum = json.loads(Path('data/corpus-lessons.json').read_text())
        sentences = [s for d in curriculum['stages'][0]['documents'] for r in d['records']
                     for s in r['sentences']][:16]
        for text in sentences:
            read(model, text)
        print(dict(sentences=len(sentences), **counts))
    finally:
        model.End()
        patch.undo()
