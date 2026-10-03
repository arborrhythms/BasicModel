"""Classify preserved failures for review; perform no rerun or source repair."""
from collections import Counter
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
live = json.loads((HERE / 'live-progress.json').read_text())
rows = []
for node, message in live['failures'].items():
    if node.startswith('test/test_basicmodel.py::') and any(
            ('::' + owner + '::') in node for owner in (
                'TestInputSpaceLexIntegration', 'TestOutputSpaceTextReconstruction',
                'TestInputSpaceTextRoundTrip', 'TestLexerConfig', 'TestXorForwardPass',
                'TestTrainEmbeddingsFlag', 'TestVocabSaveRestore')):
        category = 'retained fixture expects the retired lexicon interface'
        followup = ('Map the underlying behavior to the meronomy owner before a port '
                    'or retirement; do not restore compatibility methods to make the fixture pass.')
        if 'TestTrainEmbeddingsFlag::' in node:
            followup = ('The old Embedding type assertion blocks the optimizer-membership assertions. '
                        'Determine the live training-control contract and retain the relevant membership checks.')
        elif 'TestVocabSaveRestore::' in node:
            followup = ('The fixture cannot find the former Embedding owner, before any save/load. '
                        'Map the vocabulary/optimizer persistence assertions to the current owner; '
                        'this failure does not test the current checkpoint round trip.')
    elif node.startswith('test/test_lexicon_ownership.py::'):
        category = 'retained fixture expects the retired lexicon interface'
        followup = ('The fixture expects Embedding/getW/wv/_embed or its former checkpoint '
                    'fields, while meronomy owns a RadixLayer. Map current dictionary ownership, '
                    'optimizer membership and persistence to the live owner; retain their '
                    'behavioral assertions. A retired method-name guard is a separate retirement '
                    'decision, not a reason to remove the behavioral cases.')
    elif node.startswith('test/test_category_em_smoke.py::'):
        category = 'category role-assignment behavior; cause unresolved'
        followup = ('The codebook is enabled, role count matches and percept ids are present, '
                    'but no centroid assignment exists after fifteen forwards and hard resets. '
                    'Inspect role observations, pending learning and assignment publication. '
                    'The following nonzero-centroid-role assertion was not reached.')
    elif (node == 'test/test_meronomy_ladder.py::test_legacy_analysis_modes_are_parked_and_meronomy_is_canonical'
          or node.startswith('test/test_meronomy_ladder.py::test_undivided_legacy_modes_stage_no_spans[')):
        category = 'retained fixture expects retired reading-mode dispatch'
        followup = ('The old-mode constants and bypass dispatch were expressly retired by plan '
                    'section 14. Keep the native meronomy assertions when separating that live '
                    'contract from the old parked-mode assertions; do not restore dispatch to '
                    'satisfy a stale stub analysis_mode field.')
    elif node == 'test/test_dimensional_governance.py::test_cs_ws_recurrent_input_mismatch_raises':
        category = 'incomplete configuration fixture port'
        followup = ('The port changes MM_20M_legacy.xml to MM_20M_xor.xml but keeps the old '
                    'literal nOutput=1024 matcher; the replacement has nOutput=8. '
                    'The original HEAD fixture contains the matcher. See '
                    'dimensional-fixture-static-review.json. Preserve the intended '
                    'CS-to-WS mismatch and all validation-message assertions in a future port.')
    elif node in (
            'test/test_eval_math_thinking.py::test_eval_script_emits_every_report_column',
            'test/test_math_thinking_training.py::test_iterations_one_is_byte_identical_to_a_plain_batch'):
        category = 'input occurrence exceeds configured address capacity'
        followup = ('The same MM_math capacity exception was recorded by the configuration '
                    'first-batch audit. It occurs in _embed_radix when stamping word_offset_grid '
                    'into the input registry, before the tests reach their result assertions. '
                    'Inspect units and capacity ownership without raising the configured limit.')
    elif node == 'test/test_reconstruction_roundtrip.py::test_idempotent_config_trains_one_epoch_clean':
        category = 'input occurrence exceeds configured address capacity'
        followup = ('The full-epoch recon_bench path reaches the input-registry bound in '
                    '_embed_radix. The earlier first-batch timing completed, which does not '
                    'establish full-epoch coverage. Retain the finite-loss/version-counter '
                    'regression and inspect the staged surfaces and address units; do not '
                    'raise the configured limit to obtain a pass.')
    elif node == 'test/test_reconstruction_roundtrip.py::test_silent_zero_sites_warn_once':
        category = 'retired dead-reconstruction warning expectation'
        followup = ('The live output-shape warn-once checks pass. The failure is the second '
                    'site, which assumes mask_rate=0 and no D3 leave a dead reconstruction '
                    'channel. That warning site is absent after the reconstruction-path '
                    'retirement, and meronomy has a live perceptual reconstruction owner. '
                    'Keep the output-shape assertions; decide the second site against the '
                    'current reconstruction and unavailable-candidate reporting contract.')
    elif node == 'test/test_fineweb_preflight.py::test_checkpoint_restores_optimizer_counters_and_rng':
        category = 'resume fixture compares byte-cursor ticks with meronomy trial steps'
        followup = ('All preceding optimizer, counter and RNG restoration assertions pass. '
                    'The fixture then constructs a cursor with slab_bytes explicitly; runEpoch '
                    'uses slab_bytes=None when the resolved lexer is not byte/bytes. The migrated '
                    'MM_xor configuration uses meronomy. Compare the same cursor semantics '
                    'while retaining all persistence and resume assertions; do not change counters '
                    'merely to equal the obsolete byte-tick total.')
    elif node == 'test/test_grammar_reconstruction_gate.py::test_grammar_reconstruction_reads_the_concluded_state_before_trace_disposal':
        category = 'inverse reference-side invariant; cause unresolved'
        followup = ('The inverse spy sees reference_side[1] true during trial reconstruction, '
                    'before the commit observer checks that changing the concluded state changes '
                    'the inverse parent. Determine whether this is a valid staged object operand '
                    'or a reference-ownership regression before any assertion port. The separate '
                    'least-residual soft-gradient and single-forward read-back capture cases pass.')
    elif node == 'test/test_output_synthesis.py::test_reconstruction_and_output_are_order_independent':
        category = 'repeated-understanding reconstruction stability; cause unresolved'
        followup = ('Both within-understanding noninterference assertions pass: output does '
                    'not change reconstruction and reconstruction does not change output. '
                    'The failure compares reconstruction from two separate understand(x) calls. '
                    'Inspect admission, mutable state and forward context between readings; '
                    'do not attribute it to generation mutating one saved understanding. '
                    'Preserve the allclose assertion pending that diagnosis.')
    elif node in (
            'test/test_perceptual_loopback.py::TestPerceptStoreIntegration::test_embed_radix_respects_promotion_threshold',
            'test/test_perceptual_loopback.py::TestPerceptStoreIntegration::test_radix_percept_store_roundtrips_inserted_words'):
        category = 'promotion fixture omits separating surface bytes'
        followup = ('The hand-built _host_tokens list contains adjacent words without spaces. '
                    'With meronomy word isolation, _embed_radix joins those bytes before tiling: '
                    'hello/world becomes helloworld, and promotable/ab becomes promotableab. '
                    'The assertions then look up the unobserved separate words. Stage a valid '
                    'surface in a future port, retaining promotion thresholds, the short-word '
                    'exclusion and exact inverse-table assertions. No promotion code was changed '
                    'or rerun for this static diagnosis.')
    elif node == 'test/test_retired_names.py::test_retired_names_remain_absent':
        category = 'consolidated retired-name guard targets a removed method'
        followup = ('The source guard still resolves Spaces.PartSpace._embed, which the mode '
                    'retirement removes. Retarget each corresponding absence assertion to '
                    'its live source owner or explicitly verify the retired owner is absent. '
                    'Do not treat the unvisited entries after this exception as checked.')
    elif node.startswith('test/test_router_fires_per_word.py::test_serial_words_use_the_shared_operation_layer['):
        category = 'eager-loop fixture enters compiled reconstruction'
        followup = ('The routing fixture monkeypatches torch.while_loop to a Python '
                    'while bool(condition(...)) loop. Reconstruction through the understanding '
                    'now captures that patched loop and fails its data-dependent guard. '
                    'Use an eager boundary for this routing-subject fixture, or preserve the '
                    'real loop when testing capture; retain the operation-layer assertions. '
                    'The independent compiled-understanding and fullgraph-across-lengths '
                    'cases pass on this source.')
    else:
        category = 'awaiting cause review'
        followup = 'Keep the saved case and original assertion; inspect the failure before deciding a repair.'
    rows.append(dict(nodeid=node, category=category, followup=followup, evidence=message))
process_events = []
for path in sorted((HERE / 'full-sweep').glob('part-*/result.json')):
    part = json.loads(path.read_text())
    for worker in part['workers']:
        if worker['reason'] == 'exit':
            continue
        raw_path = Path(worker['log']).with_suffix('.json')
        raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
        node = raw.get('active')
        event = dict(nodeid=node, reason=worker['reason'],
            peak_gib=worker['peak_memory_bytes'] / 1024**3,
            worker_seconds=worker['elapsed_seconds'], log=worker['log'])
        process_events.append(event)
        if node and not any(row['nodeid'] == node for row in rows):
            rows.append(dict(nodeid=node, category='bounded worker ' + worker['reason'] + ' stop',
                followup=('The active case did not finish. Preserve the existing ceiling and '
                          'saved process record; distinguish the worker lifetime/peak from '
                          'an isolated per-case measurement. No retry is performed.'),
                evidence=event))
report = dict(stage=live['stage'], completed=live['completed'],
              category_counts=dict(Counter(r['category'] for r in rows)), cases=rows,
              process_events=process_events,
              scope='Static review of saved failures only; no new training, reruns, deletions or ports.')
(HERE / 'failure-triage.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report['category_counts']))
