"""Read the frozen first weekly record. No training, retry or source repair."""
from pathlib import Path
from collections import Counter, defaultdict
import json, subprocess
R=Path(__file__).resolve().parent; ROOT=R.parents[3]
P=ROOT/'tmp/slow-tests/20261002T012227Z-32db47'
d=json.loads((P/'record.json').read_text())
head={p:subprocess.check_output(['git','show','d679df2b:'+p],cwd=ROOT,text=True) for p in ['bin/Spaces.py','bin/Models.py','bin/Language.py','test/test_input_word_cursor.py','data/MM_20M_legacy.xml']}
evidence={}
for path,source in head.items():
 evidence[path]={term:[dict(line=i+1,text=s) for i,s in enumerate(source.splitlines()) if term in s] for term in ['analysis_store =','stage_symbolic_virgin_rows','_ws_pos_to_row','property-neutral','_stage_mixing_grammar_leaf_mask','MM_20M_legacy','<grammar','def conceptualize']}
(R/'weekly-head-static-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')

def cause(f):
 n=f['nodeid'];m=f.get('message','')
 if f['phase']=='process':return 'resource stop; origin unresolved',f["reason"]+' guard stopped the worker; no matched historical resource profile establishes when this began.'
 if "NoneType' has no len" in m:return 'recent 6.9', 'The new mixing grammar-leaf staging calls len on a missing word_texts row. The method is absent from published HEAD.'
 if "NoneType' object has no attribute 'numel'" in m:return 'recent 6.9', 'The operation-choice probe calls the new sentence-local journal writer without its global-to-local index map.'
 if 'tied reconstruction has no surface candidates' in m:return 'recent 6.9', 'The newly scoped reconstruction bank rejects a targetless sentence; the decided no-candidate rule is zero contribution plus a count.'
 if 'test_concept_readout_l1.py' in n:return 'recent 6.9', 'The probe expects batch-end proximal registration at every optimizer step; 6.9 adds earlier per-sentence trial steps.'
 if 'test_joint_objectives.py' in n:return 'recent 6.9', 'A backward probe sees loss=None at a trial step where the new projected-gradient path has already supplied gradients.'
 if 'test_prepared_answer_boundary.py' in n:return 'recent trim fixture change', 'The shared fixture replaces torch.while_loop with a Python bool loop; the test then asks the reconstruction body to compile fullgraph.'
 if 'test_reverse_traversal.py' in n:return 'recent 6.9', 'The cached normalized reconstruction cost is compared with the old eager raw/aggregate cost contract; exact source of the numerical mismatch remains unverified.'
 if 'test_available_input_targets_do_not_train' in n:return 'recent 6.9 (inferred)', 'The batch-backward observer never populates its parameter snapshot after trial-local training; the test fails on the absent snapshot, before its parameter comparison.'
 if 'test_tensor_peer_' in n:return 'recent 6.9', 'Assertions inspect the former whole-row operation trip/journal lifetime after sentence-local journal discard and closing rounds.'
 if "has no attribute 'analysis_store'" in m or "has no attribute 'stage_symbolic_virgin_rows'" in m or "has no attribute '_ws_pos_to_row'" in m:return 'long-standing retired API', 'Published HEAD already has no producer for this named WholeSpace attribute/method; see weekly-head-static-evidence.json.'
 if 'test_symbolic_iteration.py' in n:return 'long-standing retired reading path (inferred)', 'The fixture expects the old CS-leg quantizer/virgin-symbol reconstruction path; its associated methods are already absent at HEAD.'
 if 'test_dual_input_contract.py' in n:return 'long-standing reading-contract drift (inferred)', 'The assertions expect the former carrier/word-mean/quantizer route; published HEAD already labels this route property-neutral and has no analysis_store.'
 if 'test_input_word_cursor.py' in n:return 'long-standing fixture drift', 'The helper selects MM_20M_legacy, whose retained default grammar is substrate-only, but the test asserts that the configuration has a nontrivial grammar.'
 if 'test_conceptualize.py' in n:return 'long-standing shape-contract drift (inferred)', 'The test demands a three-axis shape from the conceptualize dispatch, which returns a two-axis shape. No 6.9 edit to that dispatch establishes a recent cause.'
 if 'nWhen not in (0,2)' in m:return 'long-standing shape-contract drift', 'The assertion still permits only 0/2 temporal coordinates, while the landed canonical ladder uses four.'
 if 'test_basicmodel.py::TestLoadEmbeddingsEnwiki' in n:return 'external resource', 'The optional enwiki embedding resource is absent; no embedding is loaded.'
 if 'source/.venv/bin/python' in m and 'FileNotFoundError' in m:return 'weekly snapshot harness', 'This subprocess test constructs a repository-relative .venv path, absent from the frozen source copy. Other workers use the live venv explicitly.'
 if 'test_mlx_export.py' in n:return 'stale expected failure', 'The strict xfail unexpectedly passed (XPASS); no export implementation failure was reported. The environment was not rebuilt.'
 if 'test_explicit_dimensions.py::TestXorGrammarReconstruction' in n:return 'known red gate', 'Two of four word multisets recovered. The reconstruction gate was already red in the reviewed receipt.'
 if 'test_blind_decode.py' in n:return 'unseeded quality; origin unresolved', 'Blind round-trip accuracy was .75, below 1.0. One unseeded observation does not establish a deterministic regression.'
 if 'test_serial_answer_path_varies' in n:return 'quality; origin unresolved', 'Output range .0108612 is below .1; the evidence does not separate initialization variance from a recent training change.'
 if 'test_stream_smoke.py' in n:return 'capacity; origin unresolved', 'The old embedding loader tries to insert beyond its configured 256-row capacity. No guard or configuration was raised.'
 if 'test_expectation_review.py' in n or 'test_per_word_ss_padding_noop.py' in n:return 'recent 6.9 (inferred)', 'The assertion observes a row/document STM counter after sentence commits rather than the former per-word/packed-forward lifetime.'
 if 'test_tied_reconstruction_compile.py' in n:return 'gradient reach; origin unresolved', 'The compile-count test also expects a nonzero gradient; the observed used gradients are zero. No repair or repeat was performed.'
 if 'test_word_store.py' in n:return 'retired storage contract; origin unresolved', 'The case reads former word-whole registry/trace/leaf-distill or reverse-index state. Some producers predate 6.9 retirement; this run alone does not date the specific mismatch.'
 if 'test_primed_reading_scope' in n:return 'reading scope; origin unresolved', 'The expected priming tensor is None; a precise onset is not established without another diagnostic.'
 return 'unresolved',m.splitlines()[-1] if m else 'See the raw record.'
rows=[]
for f in d['failures']:
 category,why=cause(f);rows.append(dict(nodeid=f['nodeid'],phase=f['phase'],category=category,cause=why,message=f.get('message','')))
reports={r['nodeid']:r for run in d['runs'] for w in json.loads(Path(run['result']).read_text()).get('workers',[]) for r in w.get('reports',[])}
process=[f for f in d['failures'] if f['phase']=='process']
exclusive=[f for f in process if f['nodeid'] not in reports]
counts=Counter(r['outcome'] for r in reports.values());counts['passed']+=26;counts['process_stopped_without_result']=len(exclusive)
result=dict(source_record=str(P/'record.json'),source_matched=d['source_matched'],duration_seconds=d['duration_seconds'],selected=d['selected'],unique_case_counts=dict(counts),raw_record_counts=d['counts'],guard_events=len(process),guard_events_on_cases_already_reported=[f['nodeid'] for f in process if f['nodeid'] in reports],triage=rows)
(R/'weekly-triage.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# First weekly slow run: cause triage','',f"Frozen review-13 source, matched after execution. All 328 selected cases were attempted in {d['duration_seconds']/3600:.3f} hours (302 pytest plus 26 inline). No weekly failure was repaired or rerun for this triage. Static reads of published HEAD establish absence/presence only; no HEAD measurement ran.",'',f"Unique-case accounting: {dict(counts)}. The raw record has 14 process-stop events; two occur in workers that had already reported a case outcome (one pass, one failed subtest), so adding all 14 to its 316 completed cases double-counts two. The modality test's ten failed subtests are one case. Raw outcomes remain preserved.",'','Both production native objective arms passed under their authorized 24 GiB slow ceiling; ordinary workers retained 8 GiB. The 26 inline cases passed. The CUDA exclusions and explicit MPS dispatch remain in the raw record.','', '“Inferred” and “unresolved” below are deliberate: these were not retrained on HEAD or probed by changing source. Resource stops are separate from assertion regressions.','', '| Case | Classification | Immediate cause / evidence |','|---|---|---|']
for r in rows:lines.append(f"| `{r['nodeid']}` ({r['phase']}) | {r['category']} | {r['cause']} |")
(R/'weekly-triage.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(counts),indent=2))
