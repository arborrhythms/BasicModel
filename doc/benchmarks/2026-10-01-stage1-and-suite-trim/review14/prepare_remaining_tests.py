"""Prepare explicit retirement/fixture ports; install only with --apply."""
import ast
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
changes = {}
retired = []


def get(path):
    return changes.get(path, (ROOT/path).read_text())


def save(path, source):
    ast.parse(source)
    changes[path] = source


def remove(path, names):
    source = get(path)
    lines = source.splitlines(keepends=True)
    spans = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            start = min([node.lineno]+[d.lineno for d in node.decorator_list])-1
            spans.append((start, node.end_lineno))
            if node.name.startswith('test_'):
                retired.append(dict(file=path, test=node.name,
                    old=''.join(lines[start:node.end_lineno]),
                    reason='Retired detached-student or D3 training path, plan §14.'))
    for a, b in sorted(spans, reverse=True):
        del lines[a:b]
    save(path, ''.join(lines))


remove('test/test_compiled_word_chunk.py', {
    'test_tiny_canonical_detached_reverse_stops_at_root',
    'test_tiny_canonical_detached_reverse_train_step_is_finite'})
p = 'test/test_compiled_word_chunk.py';s = get(p)
s = s.replace('detached_reverse=False, concept_rows=64', 'concept_rows=64')
a = s.index('    if detached_reverse:');b = s.index('    # BasicModel', a);s = s[:a]+s[b:]
save(p, s)
remove('test/test_detached_reverse_objective.py', {
    '_teacher_stack', 'test_reverse_construction_loss_stops_at_sentence_idea',
    'test_packed_reverse_loss_matches_serial_sentence_layout'})
p = 'test/test_detached_reverse_objective.py';save(p, get(p).replace('    ReverseConstructionChooser,\n',''))
remove('test/test_occurrence_coordinates.py', {'test_detached_word_teacher_also_excludes_field_time'})
remove('test/test_sentence_compose.py', {'test_legacy_event_reporting_does_not_zero_the_sentence_objective'})
remove('test/test_reconstruction_roundtrip.py', {'test_where_scale_applies_to_d3_reconstruction'})

p = 'test/test_concept_readout_l1.py';s = get(p)
s = s.replace('@pytest.mark.parametrize("detached_reverse", [False, True])\n','')
s = s.replace('tmp_path, monkeypatch, detached_reverse):', 'tmp_path, monkeypatch):')
s = s.replace('detached_reverse=detached_reverse, input_width=32', 'input_width=32')
save(p, s)
retired.append(dict(file=p, test='test_real_runbatch_stages_l1_once_and_reports_it_separately[True]',
    reason='Second case varied only the retired detachedReverse selection; all L1 assertions remain in the native case.'))

p = 'test/test_tied_reconstruction_objective.py';s = get(p)
s = s.replace('monkeypatch.setattr(model, "_d3_reconstruction_loss", forbidden)',
              'monkeypatch.setattr(model, "_d3_reconstruction_loss", forbidden, raising=False)')
s = s.replace('    assert training.findtext("detachedReverse") == "false"\n', '    assert training.find("detachedReverse") is None\n')
s = s.replace('''        ("<training>", "<training><teacherReconstruction>true</teacherReconstruction>"
         "<detachedReverse>true</detachedReverse>")])''',
'''        ("<training>", "<training><teacherReconstruction>true</teacherReconstruction>")])
    # Serialized keys/shapes from the current data/BasicModel.ckpt. This
    # fixture has no student forward: it exercises the one-way loader and
    # optimizer migration after the runtime's retirement.
    student = torch.nn.Module()
    shapes = {
        'idea_projection.weight': (256, 1032), 'idea_projection.bias': (256,),
        'choice_slots.weight': (775, 256), 'kind_head.weight': (3, 256),
        'kind_head.bias': (3,), 'rule_head.weight': (41, 256), 'rule_head.bias': (41,),
        'leaf_decoder.slot_e': (256, 256), 'leaf_decoder.trunk.weight': (256, 1032),
        'leaf_decoder.trunk.bias': (256,), 'leaf_decoder.out.weight': (512, 256),
        'leaf_decoder.out.bias': (512,),
    }
    for key, shape in shapes.items():
        names = key.split('.')
        parent = student
        for name in names[:-1]:
            if not hasattr(parent, name):
                parent.add_module(name, torch.nn.Module())
            parent = getattr(parent, name)
        parent.register_parameter(names[-1], torch.nn.Parameter(torch.zeros(shape)))
    legacy.symbolSpace.subspace.reverse_chooser = student
    legacy.symbolSpace.params.extend(student.parameters())''')
save(p, s)

p = 'test/test_generation_lesson.py';s = get(p)
s = s.replace('    from Models import BasicModel\n','    from Models import BasicModel\n    from Layers import Error\n')
s = s.replace('    batch, words, width = 2, 1, 3', '''    lesson_errors = {'compose': Error(), 'generate': Error()}
    lesson_errors['compose'].squared('compose', compose + 1., torch.ones_like(compose), category='expectation')
    lesson_errors['generate'].squared('generate', generate + 1., torch.ones_like(generate), category='output')
    batch, words, width = 2, 1, 3''')
s = s.replace('        _reading_lesson_reports=[], symbolSpace=SimpleNamespace(),',
              '        _grammar_lesson_errors=lesson_errors,\n        _reading_lesson_reports=[], symbolSpace=SimpleNamespace(),')
save(p, s)

p = 'test/test_compose_deadline.py';s = get(p)
s = s.replace('    from test_subspace_what_stm_contract import test_shared_operation_writes_the_caller_owned_stack',
'''    import test_subspace_what_stm_contract as contract
    original = contract._shared_compose
    def mixed_poles(model):
        state, choice = original(model)
        # Three forced NOTs leave an intersection nonzero only if the
        # operands share an occupied pole. Both binary choices remain legal.
        state[0][:, 1, 1::2].neg_()
        return state, choice
    monkeypatch.setattr(contract, '_shared_compose', mixed_poles)''')
s = s.replace('    test_shared_operation_writes_the_caller_owned_stack(_xor_model)',
              '    contract.test_shared_operation_writes_the_caller_owned_stack(_xor_model)')
save(p, s)

p = 'test/test_tied_operator_reconstruction.py';s = get(p)
s = s.replace('        torch.testing.assert_close(unknown, torch.tensor([256.]).log())',
'''        torch.testing.assert_close(unknown, torch.zeros_like(unknown))
        _assert_missing_candidate_count(bank_n, bank_bytes, torch.zeros_like(bank_valid))''')
s = s.replace('''            # Even without a candidate spelling, every promoted word remains
            # a scoreable target and pays the uniform byte/termination cost.''',
'''            # A target survives promotion even when no surface candidate is
            # available. Such a sentence is counted and contributes no term.''')
s = s.replace('            torch.testing.assert_close(cost, torch.tensor([256.]).log())',
'''            torch.testing.assert_close(cost, torch.zeros_like(cost))
            isp = model.inputSpace
            isp._ar_bank_valid.zero_()
            model._validate_reconstruction_bank()
            assert int(isp._reconstruction_missing_sentence_count) == 1
            assert not bool(isp._reconstruction_sentence_available.any())''')
s += '''

def _assert_missing_candidate_count(atoms, surfaces, valid):
    from types import SimpleNamespace
    from Models import BasicModel
    batch, candidates = atoms.shape[:2]
    isp = SimpleNamespace(
        _word_active_mask=torch.ones(batch, 1, dtype=torch.bool),
        _packed_sentence_ids=torch.zeros(batch, 1, dtype=torch.long),
        _ar_concept_lookup_rows=torch.arange(candidates)[None].expand(batch, -1),
        _ar_concept_lookup_atoms=atoms,
        _ar_concept_lookup_sentence_ids=torch.zeros(batch, candidates, dtype=torch.long),
        _ar_bank_bytes=surfaces, _ar_bank_valid=valid)
    BasicModel._validate_reconstruction_bank(SimpleNamespace(inputSpace=isp))
    assert int(isp._reconstruction_missing_sentence_count) == batch
    assert not bool(isp._reconstruction_sentence_available.any())
'''
save(p, s)

for path, source in changes.items():
    target = HERE/'remaining-preview'/path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
(HERE/'remaining-test-deletions.json').write_text(json.dumps(retired, indent=2)+'\n')
if '--apply' in sys.argv:
    for path, source in changes.items():
        (ROOT/path).write_text(source)
print('preview tests', len(changes), 'files; applied', '--apply' in sys.argv)
