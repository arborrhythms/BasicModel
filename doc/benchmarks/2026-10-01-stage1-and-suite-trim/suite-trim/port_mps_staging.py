"""The direct graph fixture must mirror runBatch's device-specific anchors."""
import json
from port_ledger import HERE,ROOT,definitions,record

file='test/test_compiled_word_chunk.py'
name='_stage_fullgraph_tensor_peer'
p=ROOT/file;s=p.read_text();old=definitions(s)[name]
new=old.replace('    slab = model.inputSpace._ar_embedded_N',
    '    slab = model.inputSpace._ar_embedded_N\n'
    '    Models._ensure_grad_anchors(slab.device, (slab.dtype,))',1)
assert new != old
(HERE/'mps-staging-anchor-repair.json').write_text(json.dumps(dict(
    failing_probes=['mps-slow-check/worker-001.log','mps-slow-check/worker-002.log'],
    reason='Direct compiled fixture bypasses runBatch, which prepares anchors before any graph. CPU fullgraph setup already did this explicitly; stage the actual tensor device/dtype here too. No production change or assertion change.',
    old=old,new=new),indent=2)+'\n')
p.write_text(s.replace(old,new,1))
record(file,name,old,[(file,name)],
       'Mirror the production eager graph boundary: prepare anchors for the staged device/dtype.',
       ['mps-staging-anchor-repair.json','mps-slow-check/worker-001.log','mps-slow-check/worker-002.log'])
print('Prepared device-specific anchors in the graph fixture; assertions unchanged.')
