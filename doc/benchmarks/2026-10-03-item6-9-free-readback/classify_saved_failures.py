"""Attach source-based diagnoses to existing outcomes; execute no tests."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE / 'failure-notes.json'
notes = json.loads(path.read_text())
files = {}
for folder in ('extra-cases/part-00', 'full-sweep/part-00'):
    for worker in json.loads((HERE / folder / 'result.json').read_text())['workers']:
        for row in worker.get('reports', []):
            if row['outcome'] in ('failed', 'error'):
                files[row['nodeid']] = str(Path(worker['log']).relative_to(HERE).with_suffix('.json'))

def add(node, classification, cause, comparison):
    notes[node] = dict(classification=classification, cause=cause, comparison=comparison,
        action='Preserve the first outcome, original assertions and budget. No repair or selected rerun after the source freeze.',
        evidence=files[node])

add('test/test_reverse_traversal.py::test_closing_chain_of_chunks_unwinds_to_the_words',
    'Remaining synthetic inverse fixture port; unwaived',
    'The fixture stages real numeric words, then substitutes unrelated random a,b,c and a+b+c without putting those vectors or their intermediate compounds in the search bank. The retired reference inverse could read them from the reference argument; free search reads the bank. The first exact-leaf assertion differs by 3.091318; later compound-routing assertions are not reached.',
    'The §20 direct helper default was witnessed reconstruction. This fixture still assumes that contract; it does not test free routing with an adequate bank.')
add('test/test_word_store.py::test_reverse_chooser_picks_fold_op_by_roundtrip',
    'Remaining operator fixture port; unwaived',
    'The fixture constructs an Ops._radmax parent and expects disjunction to recover its exact pair. Disjunction now composes the mean on every face. The name assertion passes, but one distinct expected leaf is missing.',
    'The prior §20 grammar supplied the former disjunction semantics. The current fixture still explicitly describes and constructs a radmax parent, so it is not a passing test of the new mean inverse.')
add('test/test_sparse_concept_e2e.py::test_two_phase_forward_cutover_stamps_terminal_activations',
    'Incomplete terminal-field observer port; unwaived',
    'The corrected terminal owner and shape check pass, as does the unchanged-code assertion. The next assertion expects zeroes in a physical order_slice(0) applied to the gathered field and finds eight nonzero entries. Models stamps _concept_inventory_rows; Spaces distinguishes physical order_slice from _field_order_slice. A physical dictionary slice is not generally a field-slot identity.',
    'The §20 test stopped earlier at the first-stage capacity check (15 versus the actual 8). That port is repaired, but the newly reached zero-evidence check remains unresolved. Exact field identities were not saved here; the nonzero evidence is not asserted correct or blamed on a dictionary write.')
add('test/test_reconstruction_roundtrip.py::test_mm20m_grammar_free_derivation_roundtrip',
    'Persistent free-derivation learning/recovery gap; unwaived',
    'Exact match is 0/64 after the fixed three epochs. Saved final tensors have all 346 own unit candidates and no truncation, but only 42 units recover their original surface and none of 128 whitespace positions does. Read-back picks 1 in 279 positions. Two-digit inputs already split before inversion account for 26 extra unit positions. Arithmetic on the same tensors puts 548/884 true byte/end probabilities below the existing 1e-6 clamp; see saved-free-derivation-analysis.json.',
    'Both earlier ownership rounds also observed zero exact recovery. These tensors establish repeated-digit and whitespace errors, not merely an absent measurement. They do not establish how another budget, scale, operator choice or unclipped objective would behave.')
add('test/test_byte_reconstruction.py::test_compiled_byte_reconstruction_handles_lookahead_and_nul',
    'New numerical compiled/eager parity regression; unwaived',
    'On fixed inputs, Inductor differs from eager by 4.6052039e-5 on one of two costs, above unchanged absolute 1e-5 and relative 1.3e-6 tolerances. The gradient-parity assertion is not reached. The unified scorer retains activation, so these arange-valued leaves produce much larger logits than the retired absolute-cosine branch.',
    'The §20 sweep passed. This compares the same function in eager and compiled execution on deterministic data; the exact floating-point transformation responsible has not been isolated.')
add('test/test_concept_memberships.py::test_feature_growth_preserves_frozen_rows_and_optimizer_moments',
    'Remaining optimizer-state observer port; unwaived',
    'The test indexes exp_avg for a reconstruction-owned feature parameter. Momentum SGD has no Adam exp_avg. It stops before the state-growth and frozen/live-row assertions.',
    'The §20 reconstruction optimizer was Adam. This fixture must inspect momentum state while retaining its state-preservation, enlistment and frozen-gradient assertions. No compatibility Adam state was added.')
add('test/test_grammar_word_learning.py::test_normal_text_reconstruction_updates_the_grammar_chooser',
    'New one-batch chooser-displacement failure; unwaived',
    'After the fixed unlabeled batch, operand_order.weight changed, but no mlp.* chooser parameter differs bitwise. The unchanged assertion requires an MLP update. Later enlistment, finished-state and output-policy checks are not reached.',
    'The §20 Adam case passed. Momentum SGD scales with the gradient, and the separate XOR audit shows sub-resolution gradients; however this case did not save its MLP gradients, so neither that explanation nor an ownership/enlistment defect is established.')
path.write_text(json.dumps(notes, indent=2) + '\n')
print('Saved', len(notes), 'failure classifications.')
