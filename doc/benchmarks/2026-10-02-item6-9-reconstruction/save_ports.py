"""Preserve complete before/after files, including every port's full test body."""
import ast, hashlib, json, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
REASONS={
 'configuration_fixtures.py':'Disable SBOW code pressure under §20; retain feature switches at bounded smoke-test geometry and freeze admission for repeated-state comparisons.',
 'objective_conflicts_probe.py':'Observe code geometry and VQ counts without extra forwards; rank candidates with the same free read-back scorer and honor the receipt compile setting.',
 'test_accessible_mind.py':'Supply a batch-row priming surface and boost the same symbol in that row; retrieval assertions are unchanged.',
 'test_concept_code_ownership.py':'New regressions: lookup preserves magnitude; promotion, phrase admission and quantization cannot rewrite reconstruction-owned codes.',
 'test_conceptualize.py':'Assert the current unary word definition returns distinct word/object identities and dereferences to that object; the old META triple is retired.',
 'test_existence_metadata.py':'Use the merged LTM fixture with stateless=False; scoped identity and strict checkpoint round-trip assertions remain.',
 'test_field_learning.py':'Supply the current batch-row priming tensor to the same shared-field scope check.',
 'test_free_reconstruction.py':'New regressions: signed activation × cosine × priming; gradients reach true/competitor codes; free reconstruction ignores leaf references and witness offsets.',
 'test_objective_ownership.py':'Add XOR_exact evidence coefficients to the answer-only ownership regression.',
 'test_output_path_supervised.py':'The kept ladder fixture already enables answer synthesis; assert one existing setting instead of injecting a duplicate.',
 'test_relevance_bases.py':'Inspect the terminal concept owner that receives the model\'s actual relevance integration.',
 'test_taxonomy_boundary.py':'Remove the obsolete merged-fixture import; keep the current thought fixture and all taxonomy boundary assertions.',
 'test_taxonomy_entry.py':'Remove the obsolete merged-fixture import; keep the current thought fixture and strict taxonomy checkpoint assertions.',
 'test_thought_model_fixture.py':'Build stateful thought tests from the kept LTM configuration with stateless=False.',
 'test_thought_operation_catalog.py':'The one-row thought context receives its row of the batch-local priming surface; exact values and detached storage remain checked.',
 'test_training_diagnostic_contracts.py':'The five surviving canonical configurations require context learning rate 0 and similarity scale 0 under reconstruction ownership.',
 'test_context_rotation.py':'Rotation retired by §20.3; no compatibility implementation.',
 'test_contextual_concept_codebook.py':'§20 parameter, unconstrained gradient and checkpoint contract replaces the buffer/rotation contract.',
 'test_concept_memberships.py':'Feature membership remains independent evidence; reset must not replace reconstruction-owned codes.',
 'test_row_local_concept_optimizer.py':'§20 retires post-step projection; sparse touched-row updates and untouched-row identity remain.',
 'test_inter_sentence_prediction_shape.py':'§15 detaches predictor sources and targets; predictor gradients and graph lifetime are checked.',
 'test_sentence_expectation.py':'§15 predictor-only learning with detached context and target.',
 'test_sparse_concept_e2e.py':'§20 distributional pressure is off; no SBOW gradient on codes.',
 'test_word_store.py':'Kept meronomy fixture, independent model per case, XOR smoke corpus, exact native object codes. Retired wordStore summary/WS identity tests and grammar detached-student tests removed (§15.6/§14).',
 'test_serial_object_meta.py':'Tensor peer boundaries and whitespace perception units replace the old word-only/WholeSpace.forward observers. Disabled old-mode indexing contract is retired.',
 'test_masked_semantic.py':'Use actual terminal concept owner; a grammar-free category bank has no live role centroids. Definition-evidence ablation preserves input-dependence assertion.',
 'test_dual_towers.py':'Current three-stream carrier and terminal concept owner; word reads use admitted singleton inputs; pyramid fixture supplies real edges.',
 'test_per_word_ss_padding_noop.py':'Count physical active pushes and require inactive rows to preserve every STM tensor exactly; grammar reductions retire depth=number-of-words.',
 'test_config_matrix.py':'Grammar smoke fixtures retain switches with bounded geometry; tied free inverse supplies decoded rows.',
 'test_reconstruction_roundtrip.py':'Measure registry-owned relative R/A totals instead of retired convex blend; decade interval unchanged.',
 'test_priming_energy.py':'Per-row priming and native DEF surface references replace the redundant legacy word/META bridge.',
 'test_primitive_properties.py':'Project the same exact PS/WS row sets from a native DEF record.',
 'test_conceptual_recurrence.py':'Three-stream current carrier including whole padded bind; no-recompile subject retained at smaller fixture geometry.',
 'test_frozen_concepts.py':'Compare SEEN/DESIRE at the same neutral priming state, without interleaved training changing context.',
 'test_global_attention.py':'Keep admitted vocabulary fixed when comparing a repeated read; current kept attention switches.',
 'test_global_consume.py':'Bounded kept attention configurations, fixed admission for repeated reads, current flowing subspace.',
 'test_reading_attention.py':'Bounded kept configuration; repeated-read comparison fixes admission state.',
 'test_where_attention_handoff.py':'Fix admission state for the deterministic dark-refinement comparison.',
 'test_generation_catalog.py':'Observe current owned backward with optimizer argument; answer cannot step shared operators.',
 'test_output_walk.py':'Observe current owned backward including optimizer argument and no-answer phase.',
 'test_output_synthesis.py':'Ownership audit replaces retired per-operator conflict diagnostic.',
 'test_sentence_comparison.py':'Observe each trial’s actual perception pullback gradients.',
 'test_sentence_compose.py':'Shared trial understanding in fixtures; controlled total preserves reconstruction tie and known selection.',
 'test_generation_lesson.py':'Fixture provides shared concluded trial record; weighted gradient assertion retained.',
 'test_surface_grammar.py':'Isolate supplied lesson training from reconstruction; only choosers change, never operators.',
 'test_basicmodel.py':'Native Radix surface/prototype API and explicit spans replace retired Embedding.getW fixture.',
}
RENAMES={
 'test_distributed_codes_follow_features_at_the_boundary_only':'test_definition_boundary_preserves_gradient_owned_codes',
 'test_post_step_concept_projection_is_row_local_and_signed_bounded':'test_row_local_reconstruction_step_preserves_unconstrained_codes',
 'test_prediction_trains_live_context_encoder_but_not_observed_target':'test_prediction_trains_predictor_with_detached_context_and_target',
 'test_consuming_prediction_loss_ends_encoder_graph_lifetime':'test_consuming_prediction_loss_never_retains_encoder_graph',
 'test_reconstruction_uses_category_evidence':'test_reconstruction_uses_word_definition_evidence',
 'test_full_role_loss_trains_source_and_predictor_but_never_target':'test_full_role_loss_trains_predictor_but_never_source_or_target',
 'test_conceptual_sbow_situates_live_sparse_codes':'test_distributional_code_cost_is_off_in_reconstruction_round',
 'test_contextual_codebook_is_non_grad_buffer_and_rotates_without_projection':'test_reconstruction_lookup_trains_selected_codes_without_projection',
 'test_context_owned_dictionary_is_excluded_from_optimizer_groups':'test_dictionary_is_included_once_in_optimizer_groups',
 'test_contextual_buffer_preserves_frozen_capacity_and_state_dict_storage':'test_parameter_preserves_frozen_capacity_and_state_dict_storage',
}
def tests(source):
 tree=ast.parse(source) if source else ast.Module(body=[])
 out={}
 def visit(nodes,prefix=''):
  for node in nodes:
   if isinstance(node,ast.ClassDef):visit(node.body,prefix+node.name+'::')
   elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test_'):
    start=min([node.lineno]+[d.lineno for d in node.decorator_list]);out[prefix+node.name]={'start':start,'end':node.end_lineno,'body':'\n'.join(source.splitlines()[start-1:node.end_lineno])}
 visit(tree.body);return out
manifest=[]
with zipfile.ZipFile(HERE/'before.zip') as z:
 names=set(n for n in z.namelist() if n.startswith('test/') and n.endswith('.py'))
 names.update(str(p.relative_to(ROOT)) for p in (ROOT/'test').rglob('*.py'))
 for name in sorted(names):
  old=z.read(name).decode() if name in z.namelist() else ''
  path=ROOT/name;new=path.read_text() if path.exists() else ''
  if old==new:continue
  dest=HERE/'ports'/Path(name).name;dest.mkdir(parents=True,exist_ok=True)
  (dest/'old.py.txt').write_text(old);(dest/'new.py.txt').write_text(new)
  before,after=tests(old),tests(new)
  cases=[]
  for test,body in before.items():
   target=test if test in after else RENAMES.get(test)
   if target not in after:target=None
   cases.append(dict(old=test,new=target,body_changed=target is None or body['body']!=after[target]['body'],old_lines=[body['start'],body['end']],new_lines=None if target is None else [after[target]['start'],after[target]['end']]))
  manifest.append(dict(file=name,old=str((dest/'old.py.txt').relative_to(HERE)),new=str((dest/'new.py.txt').relative_to(HERE)),reason=REASONS.get(Path(name).name,'Current owner/priming/kept-fixture contract; see implementation and focused failing probe in receipt.'),cases=cases,added=sorted(set(after)-{c['new'] for c in cases}),old_sha256=hashlib.sha256(old.encode()).hexdigest(),new_sha256=hashlib.sha256(new.encode()).hexdigest()))
(HERE/'ports.json').write_text(json.dumps(manifest,indent=2))
print(f'{len(manifest)} complete file pairs saved')
