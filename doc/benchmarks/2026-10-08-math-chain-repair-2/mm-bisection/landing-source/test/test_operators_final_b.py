"""Repair certificates: polarity, one forward snapshot, bounded affine reads."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def test_one_signless_form_per_word_and_two_independent_recovered_lanes():
    from Language import LanguageSpace, DisjunctionLayer
    from Interpret import activate_code
    from DecompositionChooser import DecompositionChooser
    op = DisjunctionLayer(); op.meaning_layout = (2, 6)
    codes = torch.tensor([[[1., 0., 1., 0., 0., 0.], [0., 1., 0., 1., 0., 0.]]])
    left_input = activate_code(codes[:, 0], torch.ones(1), 2, evidence=torch.tensor([[.7, .8]]))
    right_input = activate_code(codes[:, 1], torch.ones(1), 2, evidence=torch.tensor([[.3, .9]]))
    parent = op.compose(left_input, right_input)
    flag = torch.zeros(1, dtype=torch.bool)
    left, right, ready, details = LanguageSpace._bounded_binary_reconstruction(
        op, parent, torch.zeros_like(parent), flag, flag, codes,
        torch.ones(1, 2, dtype=torch.bool), 2, chooser=DecompositionChooser(), return_details=True)
    assert ready.all() and details['left_indices'].shape == (1, 2)
    torch.testing.assert_close(op.compose(left, right), parent, rtol=0, atol=0)
    rows = torch.tensor([[4, 5]])
    selected = int(details['selected'][0])
    n = details['right_indices'].shape[1]
    target = torch.stack((rows[0, details['left_indices'][0, selected//n]],
                          rows[0, details['right_indices'][0, selected%n]]))[None]
    loss, present, picked = DecompositionChooser.teacher_loss(details, rows, rows, target)
    assert present.all() and picked.all() and loss.isfinite().all()
    loss.sum().backward()
    assert details['relative_residual'].min() == 0


def test_unknown_alias_does_not_destroy_single_symbol_stop():
    from Language import LanguageSpace, DisjunctionLayer
    op = DisjunctionLayer(); op.meaning_layout = (2, 6)
    code = torch.tensor([[1., 0., 0., 0., 0., 0.]])
    eligible = LanguageSpace.decoder_eligibility(code, [code], [code], torch.ones(1, 2, dtype=torch.bool),
        [op], code[:, None], torch.ones(1, 1, dtype=torch.bool))
    assert eligible[0, -1]


@pytest.mark.parametrize('pair', [(1., 0.), (0., 1.), (1., 1.), (0., 0.), (.7, .8)])
def test_readback_identifies_form_independently_of_both_evidence_lanes(pair):
    from SentenceUnderstanding import readback_scores
    from Interpret import activate_code
    codes = torch.tensor([[[1., 0., 1., 0., 0., 0.], [0., 1., 0., 1., 0., 0.]]])
    leaf = activate_code(codes[:, 0], torch.ones(1), 2, evidence=torch.tensor([pair]))
    scores = readback_scores(leaf, codes, torch.ones(1, 2), meaning_start=2)
    assert scores.argmax(-1).item() == 0 and scores.shape == (1, 2)


def test_reader_block_scales_are_shared_and_preserve_additivity():
    from SentenceUnderstanding import PrimedSymbols, SentenceUnderstanding
    codes = torch.tensor([[[4., 0., 2., 0., 0., 0.], [0., 3., 0., 2., 0., 0.]],
                          [[4., 0., 2., 0., 0., 0.], [0., 3., 0., 2., 0., 0.]]])
    bank = PrimedSymbols(torch.tensor([[0,1],[0,1]]), codes, torch.ones(2,2)*2,
        torch.ones(2,2,dtype=torch.bool), torch.ones(2,2,1,dtype=torch.long),
        torch.ones(2,2,1,dtype=torch.bool), meaning_start=2, normalize_reader=True)
    a, b = codes[:,0], codes[:,1]
    torch.testing.assert_close(bank.reader_value((a+b)/2), (bank.reader_value(a)+bank.reader_value(b))/2)
    assert bank.reader_value(a)[:,:2].norm(dim=-1).max() <= 1
    assert bank.reader_value(a)[:,2:].norm(dim=-1).max() <= 1
    assert torch.equal(bank.reader_value(a)[:,:2], bank.reader_value(a*torch.tensor([1.,1.,100.,100.,100.,100.]))[:,:2])


def test_priming_has_neutral_one_and_configured_ceiling(monkeypatch):
    from Spaces import Space
    from util import TheXMLConfig
    monkeypatch.setitem(TheXMLConfig.data.setdefault('ConceptualSpace', {}), 'primingMaxBoost', 2.)
    owner = SimpleNamespace(_closed_clause_store=lambda: None)
    raw = torch.tensor([[1., 11., 6., .5], [1., 3., 2., 0.]], requires_grad=True)
    got = Space._bounded_priming(owner, raw)
    torch.testing.assert_close(got, torch.tensor([[1., 2., 1.5, .5], [1., 2., 1.5, 0.]]))
    assert not got.requires_grad
    monkeypatch.setitem(TheXMLConfig.data['ConceptualSpace'], 'primingMaxBoost', 1.25)
    assert Space._bounded_priming(owner, raw).max() == 1.25


def forced_prefix_certificate():
    """No training: the frozen 2c fixture, two readings, B and C enabled."""
    from Models import BasicModel
    from Language import OperationSelectionLayer, LanguageSpace
    from test_mm_xor import _fresh_model
    from MereologicalCodes import MereologicalCodes
    path = Path(__file__).resolve().parents[1] / 'doc/benchmarks/2026-10-06-operators-round2c/disjunction-initial-rng.pt'
    original_forward, original_attend = OperationSelectionLayer.forward, OperationSelectionLayer.attend
    commit, begin = BasicModel._commit_sentence, MereologicalCodes.begin_forward
    reads, snapshots = [], []
    def snapshot(derived):
        snapshots.append(id(derived))
        return begin(derived)
    def disjunction(module, x, **kw):
        stop=(x.shape[1]-1)*module.r_reduce+x.shape[1]*module.r_apply
        depth=kw.get('depth',torch.full((len(x),),x.shape[1]))
        kw['replay_action']=torch.where(depth>1,1,stop)
        return original_forward(module,x,**kw)
    def narrowing(module,keys,legal,space,**kw):
        action=legal.flatten(1).long().argmax(-1)
        for op in (0,2,1,4,3,5):
            slots=legal[:,:,op]
            action=torch.where(slots.any(-1),slots.long().argmax(-1)*legal.shape[-1]+op,action)
        kw['replay_action']=action
        return original_attend(module,keys,legal,space,**kw)
    def capture(m,state,sid,active,*args):
        texts,unavailable=m.reconstruct_grammar_sentence(state,sid,active)
        bank=m._sentence_primed_bank; root=state[1][9][:,sid]
        op=next(getattr(op,'gl',op) for op in m._stm_reducer().ops
                if getattr(getattr(op,'gl',op),'rule_name','')=='disjunction')
        flag=torch.zeros(len(root),dtype=torch.bool)
        left,right,valid=LanguageSpace._bounded_binary_reconstruction(op,root,
            torch.zeros_like(root),flag,flag,bank.codes,bank.valid,16)
        reads.append(dict(texts=texts, unavailable=unavailable.tolist(),
            residual=(op.compose(left,right)-root).square().sum(-1).tolist(),
            R=m._last_sentence_credit['components'][:,0,0].tolist(),
            actions=m._attention_words.actions.tolist(),
            meaning_norm=bank.codes[...,104:].norm().item(),
            form_snapshot=id(m._concept_owner().similarity_codebook.mereology._form_context)))
        return commit(m,state,sid,active,*args)
    with torch.random.fork_rng(), patch.object(OperationSelectionLayer,'forward',disjunction), \
         patch.object(OperationSelectionLayer,'attend',narrowing), patch.object(BasicModel,'_commit_sentence',capture), \
         patch.object(MereologicalCodes,'begin_forward',snapshot):
        torch.set_rng_state(torch.load(path, weights_only=True))
        model,_,data=_fresh_model('data/XOR_grammar.xml')
        try:
            derived=model._concept_owner().similarity_codebook.mereology
            assert derived.context_width == 128 and derived.symbol_centroid
            raw,target=next(iter(data.data_loader(split='test',num_streams=4)))
            batch=model.inputSpace.prepInput(raw),model.outputSpace.prepOutput(target)
            inputs=[model._bytes_to_text(x).rstrip(chr(0)) for x in raw]
            for _ in range(2):
                with torch.no_grad():
                    model.runBatch(train=False,optimizer=None,batchSize=4,split='test',batch_override=batch)
            for read in reads:
                read['multisets']=sum(sorted(a.split())==sorted(b.split()) for a,b in zip(inputs,read['texts']))
            from collections import Counter
            return dict(reads=reads, snapshots=len(snapshots), snapshot_counts=sorted(Counter(snapshots).values()), inputs=inputs)
        finally:
            model.End();model.symbolSpace.soft_reset()


def test_forced_not_and_or_descend_and_snapshot_certificate():
    report = forced_prefix_certificate()
    assert report['snapshot_counts'] == [2, 2]
    assert len(report['reads']) == 2
    assert report['reads'][1]['meaning_norm'] > 0
    for read in report['reads']:
        assert read['multisets'] == 4 and read['residual'] == [0.]*4 and read['R'] == [0.]*4


def test_both_training_trials_share_bootstrap_snapshot():
    from test_mm_xor import _fresh_model
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    try:
        optimizer = model.getOptimizer(lr=.01)
        for _ in range(3):
            model.runEpoch(optimizer=optimizer, batchSize=4, split='train', max_batches=1)
            for audit in model._sentence_credit_audits:
                assert not audit['components'][..., 0].any()
        assert model._concept_owner().similarity_codebook.mereology.context_audit()
    finally:
        model.End(); model.symbolSpace.soft_reset()
