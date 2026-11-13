"""Probe input boundaries use frozen source pairs without admitting their targets."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS


def test_production_probe_uses_only_sources_and_one_runtime_document_per_pair():
    from eval_games import game_rows, production_probe
    root=Path(__file__).resolve().parents[1]
    rows=game_rows(root/'data/games')['conjunction'][:2]
    assert len(rows)==2
    staged=[];calls=[];resets=[]
    @contextmanager
    def runtime_batch(inputs, *, document_keys):
        staged.append((list(inputs),document_keys))
        yield
    model=NS(spaces=[],conceptualSpaces=[],
        inputSpace=NS(data=NS(runtime_batch=runtime_batch),prepInput=lambda values:values),
        outputSpace=NS(prepOutput=lambda values:values),
        flush_word_buffers=lambda:None,dispatch_per_row_reset=resets.append,
        dispatch_soft_reset=lambda:None,post_tick_compact=lambda:None)
    def run(**options):
        calls.append(options)
        model._last_answer_construction=NS(texts=[rows[0]['target']] if
            len(staged)<=len(rows[0]['sources']) else [])
    model.runBatch=run
    reading=production_probe(model,rows)
    expected=[([source],[('game',row['key'])]) for row in rows for source in row['sources']]
    assert staged==expected
    assert all(not call['train'] and call['split']=='runtime' for call in calls)
    assert all(question.split=='runtime' for call in calls for question in call['questions'])
    assert sum(bool(call['questions']) for call in calls)==2
    assert sum(value==[True] for value in resets)==2
    assert reading['count']==2 and reading['correct']==1 and reading['value']==.5
