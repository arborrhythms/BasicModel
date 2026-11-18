"""Observers preserve native results and reject unequal inventory receipts."""
import json
from copy import deepcopy
import pytest
from sentence_host_inventory import compare_states


def test_inventory_comparison_checks_optimizer_and_stream(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    with pytest.raises(AssertionError,match='no observed'):compare_states(a,b)
    record=dict(label='sentence',parameters={'x':1},optimizer={'x':2},stream={'x':3})
    for p in (a,b):(p/'batch-000.json').write_text(json.dumps(record))
    assert compare_states(a,b)==1
    for key in ('parameters','optimizer','stream'):
        bad=deepcopy(record);bad[key]={'x':4}
        (a/'batch-000.json').write_text(json.dumps(bad))
        with pytest.raises(AssertionError,match=key):compare_states(a,b)


def test_stepped_trace_and_provenance_preserve_read_and_state(tmp_path,monkeypatch,eager_reading):
    import Assessment
    from postmerge_word_lanes import stepped_trace,native_provenance,word_deposits
    from test_assessment import snapshot,equal
    from test_compiled_word_chunk import _tiny_canonical_model
    model=_tiny_canonical_model(tmp_path,monkeypatch,batch_size=1)
    model._tensor_peer_while_eager=True
    try:
        model.understand(model.inputSpace.prepInput(['alpha beta']));model.End()
        before=snapshot(model)
        plain=Assessment.read_sentence(model,'alpha novel',trace=True)
        records=[]; deposits=[]
        with monkeypatch.context() as patch:
            patch.setattr(Assessment,'_attention_trace',stepped_trace)
            with native_provenance(records.append), word_deposits(deposits.append):
                observed=Assessment.read_sentence(model,'alpha novel',trace=True)
        assert records and records[-1]['words']
        assert any(row['kind']=='deposit' and row['count'] for row in deposits)
        assert all(not row['negative'] for row in deposits if row['kind']=='deposit')
        trace=observed['attention']['trace']
        assert trace and trace['masks'] and trace['choices']
        assert trace['actions'][0]==observed['attention']['actions']
        observed['attention']['trace']=plain['attention']['trace']
        assert observed==plain
        equal(snapshot(model),before)
    finally:
        model.End();model.symbolSpace.soft_reset()


def test_size_observer_keeps_maximum_and_counts_only_targets():
    from host_object_inventory import Sizes
    sizes=Sizes([])
    key=('fixture','read','menu')
    sizes.size(key,[1,2,3]);sizes.size(key,[1]);sizes.size(key,[1,2,3,4])
    assert sizes.values[key]==dict(kind='list',size=4)
    sizes.size(('fixture','read','opaque'),object())
    assert len(sizes.values)==1


def test_idea_read_observer_compares_both_readers_on_the_same_joined_field():
    from types import SimpleNamespace
    import torch
    from postmerge_word_lanes import idea_read_measurements
    codes = torch.tensor([[1., 0., .2, .4], [0., 1., .8, .6]])
    owner = SimpleNamespace(_order_caps=lambda: (2,),
        _expected_knowing=torch.zeros(1, 2, 2),
        concept_id_at_row=lambda row: row+1, _csw_row_of=lambda cid: cid-1,
        _row_order=lambda row: 0,
        idea_stack=lambda value: value,
        similarity_codebook=SimpleNamespace(lookup_rows=lambda rows: codes[rows],
            mereology=SimpleNamespace(percept_event_width=2)))
    observed = torch.tensor([[[.8, .2], [.4, .1]]])
    expected = observed*.5
    target = torch.tensor([[.6, .4, .3, .5]])
    report = idea_read_measurements(owner, [dict(presentation=1, observed=observed,
        expected=expected, target=target, no_omission=True)])
    row, = report['records']
    centroid = (.8*codes[0]+.4*codes[1])/1.2
    for name, region in (('form',slice(0,2)),('meaning',slice(2,None))):
        assert row['blocks'][name]['centroid_mse'] == pytest.approx(
            float((centroid[region]-target[0,region]).square().mean()))
        assert row['blocks'][name]['read_mse'] == pytest.approx(
            float((observed.flatten(1)[0,region]-target[0,region]).square().mean()))
    assert row['point_mse'] == row['field_mse']
    assert not owner._expected_knowing.any()


def test_report_uses_measured_commit_and_comparison(tmp_path,monkeypatch):
    import attention_host_report
    source=tmp_path/'source';source.mkdir()
    measured=tmp_path/'inventory.json'
    measured.write_text(json.dumps(dict(source_commit='measured-commit',equal_batches=16,
        records=[dict(label=dict(texts=['measured sentence']))],counts={},host_sites=[],loops=[])))
    target=tmp_path/'report'
    monkeypatch.setattr('sys.argv',['report','--source',str(source),'--inventory',str(measured),'--output',str(target)])
    attention_host_report.main()
    report=json.loads(target.with_suffix('.json').read_text())
    assert report['source_commit']=='measured-commit' and report['equal_batches']==16
    assert report['sentence']=='measured sentence'
    assert '16 observed batches' in target.with_suffix('.md').read_text()
