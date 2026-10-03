"""Retire legacy APIs with no production/configuration/grammar/checkpoint use."""
import ast,json,re,textwrap
from port_ledger import ROOT,HERE,definitions,record,remove_definitions
EVIDENCE=['legacy-path-use-audit.json','legacy-checkpoint-string-audit.json','remaining-legacy-entrypoint-uses.txt']
changes=[]
def write(path,new):
 p=ROOT/path;old=p.read_text();ast.parse(new) if p.suffix=='.py' else None
 if old!=new:changes.append(dict(file=path,old=old,new=new));p.write_text(new)
def retire(filename,names,reason):
 old=remove_definitions(filename,names)
 for name,body in old.items():
  tests=[(q,b) for q,b in definitions(textwrap.dedent(body)).items() if q.split('::')[-1].startswith('test_')]
  if name.startswith('test_') or '::test_' in name:tests=[(name,body)]
  for q,b in tests:record(filename,q,b,[],reason,EVIDENCE)
 return old
# Discourse aliases are unused; predict() remains a live Models.py call.
# The first run removed these three methods before a receipt-helper indentation error.
# Their complete old bodies remain in pre-trim-source.zip.
assert 'InterSentenceLayer::snapshot' not in definitions((ROOT/'bin/Layers.py').read_text())
retire('test/test_discourse_space.py',['TestBackCompatShims::test_snapshot_alias_calls_observe','TestBackCompatShims::test_contrastive_loss_alias_returns_arma_mse','TestBackCompatShims::test_predictive_loss_alias_is_no_op'],'Remove only the three unused discourse aliases and their tests; observe and predict stay live.')
# Ops' old inverted-polarity positional API and single-operand analytic inverses.
s=(ROOT/'bin/Layers.py').read_text()
s=s.replace('_NO_MODE = object()\n','')
for name,mode in (('_lift_kernel','OR'),('_lower_kernel','AND')):
 s=s.replace(f'def {name}(X1, X2=None, mode=_NO_MODE,',f'def {name}(X1, X2=None, mode={mode!r},')
 start=s.index('        Legacy positional form Ops.'+name)
 end=s.index('        if mode ==',start)
 s=s[:start]+'        """\n'+s[end:]
start=s.index('        This is the new multi-return convention;')
end=s.index('        ``left_rows``',start)
s=s[:start]+s[end:]
write('bin/Layers.py',s)
retire('bin/Layers.py',['Ops::liftReverse','Ops::lowerReverse'],'Unused two-argument analytic inverse; live basis-backed pair inverses remain.')
retire('test/test_ops_lift_lower.py',['TestDeprecationAliases'],'Unused positional polarity aliases and two-argument analytic inverses retired; all explicit-mode operator tests remain.')
retire('test/test_stm_reverse_roundtrip_lift_lower.py',['TestSmoothAnalyticReverse'],'Only the removed two-argument analytic inverse path is exercised; basis-backed reverse and all retained tolerances remain.')
p=ROOT/'test/test_ops_lift_lower.py';s=p.read_text();s=s.replace('    - legacy positional 2-arg form fires DeprecationWarning and returns\n      bit-exact pre-refactor outputs (smoothed body)\n','');write(str(p.relative_to(ROOT)),s)
# Lexicon's ball=False torus geometry is used by _SBOWModule; only the alias goes.
s=(ROOT/'bin/Layers.py').read_text().replace('*, ball: bool = True, torus: Optional[bool] = None,','*, ball: bool = True,')
a=s.index('        # geometry. ``torus=``');b=s.index('        self.ball = bool(ball)',a)
s=s[:a]+'        # geometry. Callers select it explicitly with ``ball=False``.\n'+s[b:]
a=s.index('    # Backward-compat property: ``self.torus``');b=s.index('    # -- Unit-ball geometry',a)
s=s[:a]+s[b:];write('bin/Layers.py',s)
retire('test/test_lexicon_unit_ball.py',['TestTorusLegacyMode::test_legacy_torus_kwarg_alias'],'Unused torus keyword/property aliases removed; ball=False geometry remains in production SBOW.')
p=ROOT/'test/test_lexicon_unit_ball.py';s=p.read_text();old=definitions(s)['TestTorusLegacyMode::test_torus_init_inside_canonical_cell'];s=s.replace('        self.assertTrue(emb.torus)               # backward-compat alias\n','');write(str(p.relative_to(ROOT)),s)
record(str(p.relative_to(ROOT)),'TestTorusLegacyMode::test_torus_init_inside_canonical_cell',old,[(str(p.relative_to(ROOT)),'TestTorusLegacyMode::test_torus_init_inside_canonical_cell')],'Remove only the assertion on the retired torus property; every canonical geometry assertion stays.',EVIDENCE)
# Retired range-encoding D keyword: only test callers, no live configuration key.
s=(ROOT/'bin/Spaces.py').read_text();a=s.index('# ``_WHEN_TENSE_DEFAULT``');b=s.index('# Tense TIME step',a);s=s[:a]+s[b:]
s=s.replace('def encode(self, start, end=None, D=None):','def encode(self, start, end=None):')
s=s.replace('magnitude-D tense axis is retired (the ``D`` kwarg on ``encode`` is accepted\n    for back-compat and ignored). nDim=2 (== nWhere); disabled (nDim=0) when','magnitude-D tense axis is retired. nDim=2 (== nWhere); disabled (nDim=0) when')
s=s.replace('duration: angle = time center, magnitude = duration. ``D`` is accepted\n        for back-compat with the retired magnitude-tense scheme and IGNORED.','duration: angle = time center, magnitude = duration.')
write('bin/Spaces.py',s)
retire('test/test_when_range_encoding.py',['test_back_compat_D_kwarg_is_ignored'],'The unused magnitude-tense D keyword is retired; temporal range behavior stays.')
for file in ('test/test_modality_widths.py','test/test_model_time_when.py','test/test_grammar_fixtures.py'):
 p=ROOT/file;s=p.read_text();olddefs=definitions(s);new=s.replace('_WHEN_TENSE_DEFAULT, ','').replace('encode(0, D=_WHEN_TENSE_DEFAULT)','encode(0)');write(file,new)
 for name,old in olddefs.items():
  if name.split('::')[-1].startswith('test_') and old!=definitions(new).get(name):record(file,name,old,[(file,name)],'Use canonical instant encoding; the retired ignored D argument changes no expected tensor or assertion.',EVIDENCE)
# Remove the reverseScale XML rename; no checked-in config contains the old key.
s=(ROOT/'bin/util.py').read_text();a=s.index('        Currently handles ``<reverseScale>``');b=s.index('        architecture =',a);s=s[:a]+('        Retired architecture knobs are rejected at ingestion.\n        """\n')+s[b:]
a=s.index('        training = architecture.get("training", {})',a);b=s.index('\n    def reload(',a);s=s[:a]+s[b:];write('bin/util.py',s)
p=ROOT/'data/model.xsd';s=p.read_text();s=s.replace('      <xs:element name="reverseScale" type="unitInterval" minOccurs="0"/>\n','');s=re.sub(r'           Legacy name reverseScale still parsed.*?\n', '', s);write('data/model.xsd',s)
retire('test/test_ir_only_refactor.py',['TestReverseScaleBackcompat::test_reverse_scale_legacy_maps_through'],'No active configuration uses reverseScale; the canonical reconstructionScale parser test remains.')
p=ROOT/'test/test_ir_only_refactor.py';s=p.read_text();a=s.index('    """Legacy ``<reverseScale>``');b=s.index('    """',a+7)+7;s=s[:a]+'    """The canonical reconstruction cost weight parses without a warning."""'+s[b:];write(str(p.relative_to(ROOT)),s)
# Relative truth requires a declared named start or a real relative operator.
p=ROOT/'bin/Language.py';s=p.read_text();a=s.index('    def _relative_start_categories(');b=s.index('    def _relative_rule_id_set(',a)
s=s[:a]+'''    def _relative_start_categories(self):
        """Named relative-truth starts retained from the grammar declaration."""
        return set(self.ws_relative_starts or ())

'''+s[b:];write(str(p.relative_to(ROOT)),s)
retire('test/test_relative_rule_detection_collapsed.py',['test_rel_t_back_compat_fallback'],'No active grammar needs an unnamed REL_T fallback; transitional_pos already names relative_truth.')
# Retire the uncalled snapshot adapters while retaining live codebook capabilities.
retire('bin/space_carrier.py',['SpaceCarrierMixin::carrier_schema','SpaceCarrierMixin::_carrier_activation','SpaceCarrierMixin::_carrier_band_from_index','SpaceCarrierMixin::to_pipeline_carrier','SpaceCarrierMixin::from_pipeline_carrier','SpaceCarrierMixin::as_pipeline_stage'],'No production caller of these completed migration adapters.')
p=ROOT/'bin/space_carrier.py';s=p.read_text();b=s.index('from __future__');s='''"""Read-only codebook capabilities and ownership versions for pipeline spaces.

Pipeline stages consume immutable SubSpace carriers directly. The completed
legacy snapshot conversion adapters were retired in the October 1 suite trim.
"""

'''+s[b:];s=s.replace('from util import TheDevice\n','');write(str(p.relative_to(ROOT)),s)
retire('test/test_space_carrier_bridge.py',[
 'test_space_exports_sparse_selection_with_live_read_only_identity',
 'test_legacy_spaces_pipeline_via_fresh_snapshot_adapters',
 'test_legacy_stage_requires_explicit_training_safety_migration',
 'test_legacy_stage_requires_an_explicit_pipeline_safety_audit',
 'test_dense_carrier_round_trips_through_unowned_legacy_input_adapter',
 'test_prior_trace_and_typed_mutations_survive_legacy_snapshot'],
 'Tests require the unused Space-to-carrier snapshot adapter. The live capability/version assertion path stays in test_executor_advances_bound_space_version_and_invalidates_replicas; checkpoint loading stays in test_legacy_duplicate_checkpoint_keys_migrate_to_single_owners.')
# Complete file snapshots supplement the individual old/new test bodies.
(HERE/'legacy-path-source-changes.json').write_text(json.dumps(changes,indent=2)+'\n')
print('Audited unused discourse, operator, alias, range, XML, grammar-fallback and snapshot-adapter paths removed.')
