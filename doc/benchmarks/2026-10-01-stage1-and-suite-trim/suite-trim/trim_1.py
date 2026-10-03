"""Apply the audited precursor retirement after the live byte ports pass."""
import json
from pathlib import Path
from port_ledger import ROOT, HERE, definitions, record, remove_definitions
assert json.loads((HERE/'utf8-port-byte-boundary-check/result.json').read_text())['exit_code'] == 0
mapping=json.loads((HERE/'precursor-coverage-map.json').read_text())
old_guard=definitions((ROOT/'test/test_ps_reverse_e2e.py').read_text())['test_grammar_has_no_and_token']
(ROOT/'test/test_retired_names.py').write_text('"""Retired names stay absent from the live architecture."""\n\n'+old_guard+'\n')
ports={
 'test_routed_byte_fallback_round_trips_non_ascii':'test_byte_witness_round_trips_non_ascii',
 'test_routed_byte_terminals_preserve_offsets_and_dont_overlap':'test_byte_witness_offsets_cover_the_surface_without_overlap',
 'test_split_multibyte_glyph_has_no_replacement_char':'test_split_multibyte_witness_keeps_raw_bytes',
 'test_known_word_with_byte_fallback_mix_round_trips':'test_admitted_word_and_utf8_bytes_replay_together',
 'test_meronymic_reverse_replays_surface':'test_surface_witness_replays_words_and_spaces',
 'test_meronymic_synthesis_can_emit_spaces':'test_surface_witness_replays_words_and_spaces',
 'test_routed_analysis_round_trips':'test_surface_witness_replays_words_and_spaces',
 'test_ps_analyzer_byte_fallback':'test_unfamiliar_word_keeps_every_nonzero_byte_row',
}
for entry in mapping['tests']:
 filename,name=entry['id'].split('::',1)
 new=[]
 if name in ports:
  new=[('test/test_meronomy_utf8.py',ports[name])]
  entry['action']='ported original behavior to the live eager byte boundary'
  entry['live_coverage']=[f'{a}::{b}' for a,b in new]
 elif name=='test_grammar_has_no_and_token':new=[('test/test_retired_names.py',name)]
 entry['new_bodies']=[{'id':f'{a}::{b}','body':definitions((ROOT/a).read_text())[b]} for a,b in new]
 record(filename,name,entry['old_body'],new,entry['action']+'; '+entry['decision'],
        ['precursor-and-dead-code-use-audit.json',*entry['live_coverage']])
files={r['id'].split('::')[0] for r in mapping['tests']}
for filename in files:
 if filename.endswith('test_within_whole_division.py'):
  names=[r['id'].split('::')[1] for r in mapping['tests'] if r['id'].startswith(filename+'::')]
  remove_definitions(filename,names+['_oss','_terminals'])
  p=ROOT/filename;s=p.read_text();a=s.index('Two seams are covered:');b=s.index('"""',a)
  s=s[:a]+('The live seam is ``WholeSpace.stage_analysis_spans`` ->\n'
            '``Spaces._divide_spans_into_attested``; attestation comes from the peer\n'
            'percept store and ``<WholeSpace><divideWithinWhole>`` defaults on.\n')+s[b:];p.write_text(s)
 else:(ROOT/filename).unlink()
(ROOT/'bin/perceptual_analyzer.py').unlink()
mapping['status']='Applied after all eight live byte-witness port cases passed; no production importer or checkpoint reference found.'
(HERE/'precursor-coverage-map.json').write_text(json.dumps(mapping,indent=2)+'\n')
# Comments only: the standalone prototype no longer defines the live boundary.
p=ROOT/'bin/Language.py';s=p.read_text().replace('# (the gradient analogue of\n# ``perceptual_analyzer.soft_operator_compose``) plus an MSE loss', '# plus an MSE loss').replace('Tensor-weighted operator superposition over ``op_names`` -- the\n    differentiable analogue of ``perceptual_analyzer.soft_operator_compose``.', 'Tensor-weighted connective supervision over ``op_names``.').replace('(gradient\n    flows to it, unlike the float-coerced ``soft_operator_compose``). A','(gradients\n    flow to these supervision weights). A');p.write_text(s)
p=ROOT/'bin/Spaces.py';s=p.read_text().replace('This is the invertible EndpointSumWhere form (perceptual_analyzer.py)\n# adopted into the muxed event tail so ``.where`` / ``.when`` carry the start-end','This endpoint-sum helper remains for range encodings; the live\n# WhereEncoding below uses the start only. A range codec carries the start-end').replace('retired from the muxed band (the analyzer\'s ``EndpointSumWhere`` span key\n    in perceptual_analyzer.py is a different codec and keeps it).','retired from the muxed band. The standalone analyzer using that span key\n    was retired with its separate routing protocol.').replace('This mirrors the codebase\'s sub-half-period span convention\n        (``perceptual_analyzer.EndpointSumWhere.div_term = pi/(2*namespace)``).','The sub-half-period convention uses ``pi/(2*namespace)``.');p.write_text(s)
p=ROOT/'bin/architecture.py';s=p.read_text().replace('retired from the muxed band (the analyzer\'s EndpointSumWhere keeps it).','retired from the muxed band; the standalone endpoint-sum analyzer is retired too.');p.write_text(s)
p=ROOT/'test/test_where_bracket.py';s=p.read_text().replace('``EndpointSumWhere`` span key (perceptual_analyzer.py) is a separate codec and','retired standalone ``EndpointSumWhere`` span key was a separate codec and');p.write_text(s)
p=ROOT/'doc/Mereology.md';s=p.read_text();a=s.index('`MeronymicRouter` in',s.index('### Live Routing and Implementation Boundary'));b=s.index('The live router and span bookkeeping',a)
s=s[:a]+'''The live reading boundary is `Models._lex_embed_stem`: InputSpace supplies the
byte slab and WholeSpace stages the property-defined unit spans; PartSpace's
meronomy ladder retains every unit's ordered byte witness and admits recurring
units into the percept store. The compiled word loop consumes the staged part
IDs, masks and offsets. Grammar operations consume object references above this
boundary; a white-space unit remains in perception and byte reconstruction.

The standalone `perceptual_analyzer.py` prototype, including its cosine-based
merge router and endpoint-sum span codec, was removed in the October 1 suite
trim. It is not the live routing implementation. Its exact byte-replay behavior
is covered by `test/test_meronomy_utf8.py` and `test/test_meronomy_ladder.py`;
within-whole division is covered on WholeSpace by
`test/test_within_whole_division.py`. The live start-only WhereEncoding and its
`test/test_where_bracket.py` tests remain. The use and coverage audit is in
[the trim receipt](benchmarks/2026-10-01-stage1-and-suite-trim/suite-trim/precursor-coverage-map.json).

'''+s[b:];s=s.replace('The live router and span bookkeeping establish the path and the complete order\nspecification.', 'The live span bookkeeping and ordered byte witness establish the path and order\nspecification.');p.write_text(s)
print('Retired standalone analyzer and 34 cases; eight live port cases passed; WhereEncoding tests retained.')
