"""Ports of D3 callers to retained owners. Full old/new bodies are archived."""
import ast,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[3]
changes={}
def get(p):
    previous=R/'remaining-preview'/p
    return previous.read_text() if previous.exists() else (ROOT/p).read_text()
def save(p,s):
    ast.parse(s)
    changes[p]=s

p='test/test_occurrence_coordinates.py';s=get(p)
s=s.replace('def test_d3_scores_content_and_position_without_repeated_time_credit():','def test_event_inverse_scores_content_and_position_without_repeated_time_credit():')
s=s.replace('    from Layers import ModelLoss\n','    from Layers import ModelLoss, Error\n')
s=s.replace('''        perceptualSpace=None, loss=ModelLoss(),
        _stm_single_S=torch.zeros(1, 16), _reverse_from_S=lambda _: pred)''','''        perceptualSpace=None, loss=ModelLoss())''')
s=s.replace('    loss, metric = BasicModel._d3_reconstruction_loss(owner)','''    errors = Error()
    loss = owner._reverse_event_loss(pred, target, score_when=False, registry=errors)
    metric = errors.breakdown()''')
s=s.replace('    again, metric_again = BasicModel._d3_reconstruction_loss(owner)','''    later = Error()
    again = owner._reverse_event_loss(pred, target, score_when=False, registry=later)
    metric_again = later.breakdown()''')
s=s.replace('    torch.testing.assert_close(metric_again, metric)','    assert metric_again == metric')
save(p,s)
p='test/test_reconstruction_roundtrip.py';s=get(p)
s=s.replace('    assert model._d3_active, "grammar train batch should take the D3 path"','''    assert model.reconstruct_in_loop, "a grammar reading owns its tied byte objective"
    assert model._last_understanding.input_reconstruction is not None''')
s=s.replace('on the serial/D3 path the two','on the former serial/D3 path the two')
save(p,s)
p='test/test_masked_semantic.py';s=get(p)
s=s.replace('''    assert rec["model"]._d3_active is False, (
        "D3 per-word objective must not displace the whole-slab masked-LM")''','''    assert rec["model"].reconstruct_in_loop is False, (
        "a grammar-free reading keeps the whole-slab masked-LM objective")''')
save(p,s)
p='bin/bench_training_step.py';s=get(p)
s=s.replace('''"d3_active": bool(getattr(self, "_d3_active", False)),
                     "detached_reverse": bool(getattr(self, "detached_reverse", False))''','''"reconstruct_in_loop": self.reconstruct_in_loop,
                     "reconstruction_scope": self.reconstruction_scope''')
save(p,s)
p='test/test_aligned_fold_binding.py';s=get(p)
s=s.replace('recon = model._reverse_from_S(model._stm_single_S)','recon, _ = model.reverseReconstruct(model._capture_understanding(result), train=True)')
save(p,s)
p='test/test_stm_recon_from_cleared_cache.py';s=get(p)
s=s.replace('            model.forward(x)','            result = model.forward(x)\n            model._test_understanding = model._capture_understanding(result)')
s=s.replace('cb = ps._mphf_codebook() if ps is not None else None','cb = ps._basis if ps is not None else None')
s=s.replace('recon = model._reverse_from_S(S)','recon, _ = model.reverseReconstruct(model._test_understanding)')
s=s.replace('the MPHF word codebook', 'the native percept codebook').replace('the MPHF codebook','the native percept codebook')
# Keep the historical finding and its unchanged .8 overlap criterion. Only
# the active API description changes; the xfail still reports the real result.
s=s.replace('''    """After clearing the cache, ``_reverse_from_S(S)`` runs from S ALONE''','''    """After clearing the cache, reverseReconstruct reads its owned understanding''')
save(p,s)
p='test/test_grammar_separator.py';s=get(p)
a=s.index('    event_loss = model._reverse_event_loss');b=s.index('    try:',a)
s=s[:a]+'''    word_cost = model._byte_word_cost
    def observed_cost(idea, word, bank_n, bank_bytes, bank_valid,
                      target_bytes, target_valid, ready):
        targets.append(target_bytes.detach().clone())
        return word_cost(idea, word, bank_n, bank_bytes, bank_valid,
                         target_bytes, target_valid, ready)
    monkeypatch.setattr(model, '_byte_word_cost', observed_cost)
'''+s[b:]
s=s.replace('            model._d3_reconstruction_loss()\n','')
s=s.replace('''        assert targets and torch.equal(targets[-1], isp._ar_embedded)''','''        assert targets and torch.equal(targets[-1], isp._ar_target_word_bytes)
        for b in range(4):
            assert bytes(targets[-1][b, 1][isp._ar_target_word_mask[b, 1]].tolist()) == b' '
        assert model._capture_understanding(model._last_execution).input_reconstruction is not None''') if False else s.replace('''        assert targets and torch.equal(targets[-1], isp._ar_embedded)''','''        assert targets and torch.equal(targets[-1], isp._ar_target_word_bytes)
        for b in range(4):
            assert bytes(targets[-1][b, 1][isp._ar_target_word_mask[b, 1]].tolist()) == b' ' ''')
save(p,s)
p='test/test_relative_error.py';s=get(p)
s+='''\n\ndef test_zero_weight_does_not_read_a_missing_registry():
    errors = Error()
    errors.merge(object(), weight=0.)
    assert errors.total() is None
    assert errors.breakdown() == {}
'''
save(p,s)
p='test/test_train_compile_target.py';s=get(p)
s=s.replace('"MM_20M_legacy.pte"','"MM_20M_xor.pte"')
save(p,s)
for p in ['bin/bm.py','bin/export_mlx.py']:
    save(p,get(p).replace('MM_20M_legacy.xml','MM_20M_xor.xml'))
p='test/test_recon_bench.py';s=get(p)
s=s.replace('MM_20M_legacy (byte cursor) decodes via the _last_host_slab stash.',
            'The native XOR configuration decodes through the receipt harness.')
save(p,s)
# The packing helper existed solely to feed the retired D3 comparison.
p='test/test_reverse_left_shift.py'
retired=dict(file=p,reason='Only caller of the removed D3-only packing helper; tied reconstruction owns its sentence layout.',old=(ROOT/p).read_text())
(R/'extra-port-deletions.json').write_text(json.dumps([retired],indent=2)+'\n')
for p,s in changes.items():
    target=R/'extra-preview'/p;target.parent.mkdir(exist_ok=True,parents=True);target.write_text(s)
if '--apply' in sys.argv:
    for p,s in changes.items():(ROOT/p).write_text(s)
    (ROOT/retired['file']).unlink()
print('extra preview',len(changes),'applied','--apply' in sys.argv)
