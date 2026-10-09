"""Preserve the running dry revision from its saved hashes, without editing it."""
from pathlib import Path
import hashlib,json,zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
expected=json.loads((HERE/'development-epoch-answer-02/source.json').read_text())
files={name:(ROOT/name).read_text() for name in expected}
def undo(name, new, old):
    assert new in files[name], (name,new[:80])
    files[name]=files[name].replace(new,old)
undo('bin/Models.py','def exhausted(current, reason="work_budget"):', 'def exhausted(current):')
undo('bin/Models.py','"incomplete": (reason,),','"incomplete": ("work_budget",),')
undo('bin/Models.py','if (reason == "work_budget" and state is not None and not state.forced\n                    and meter.spent > recorded_spend):','if state is not None and not state.forced and meter.spent > recorded_spend:')
undo('bin/Models.py', '''                    if not actions:
                        # With no legal operation there is no further work.
                        # Finish with the reference open, without concluding it.
                        outcome = exhausted(value, reason='no_legal_operation')
                        if not parents:
                            return outcome
                        outcome = return_child(parents.pop(), outcome)
                        current = outcome['meaning']
                        continue
''','')
undo('bin/Models.py', '''                if self._sentence_departure is not None:
                    sampled = self._sentence_departure['compose_round']
                    departure = torch.where(sampled >= 0, sampled, departure)''', '''                sampled = self._sentence_departure['compose_round']
                departure = torch.where(sampled >= 0, sampled, departure)''')
undo('bin/SentenceCredit.py', '''        if bool(compose.any()):
            raise ValueError('compose departure requires its reservoir snapshot round')
        compose_round = torch.full_like(walk, -1)''', '        compose_round = departure_at(compose)')
undo('bin/WalkTrials.py', '''    if eligible.shape[-1] == 0:
        return torch.full((eligible.shape[0],), -1, device=eligible.device, dtype=torch.long)
''','')
undo('test/test_grammar_word_learning.py', '        from compose_score_probe import observe_score_function', '''        receipt = _ROOT / 'doc/benchmarks/2026-10-06-operators-round2'
        monkeypatch.syspath_prepend(str(_ROOT / 'doc/benchmarks/2026-10-03-operators-attention'))
        monkeypatch.syspath_prepend(str(receipt))
        from round2_score_probe import observe_score_function''')
s=files['test/test_math_chain_repair2.py'];start=s.index('def test_no_legal_thought_operation');end=s.index('def test_rebuilt_reference_bank',start);files['test/test_math_chain_repair2.py']=s[:start]+s[end:]
undo('test/test_negative_expectation.py', '''def test_expectation_updates_predictors_outside_the_departure_comparison(monkeypatch):
    """Expectation trains its predictor; R + A judges the departure."""
    from test_item6_2_thinking import test_expectation_trains_predictor_without_moving_departure_chooser
    test_expectation_trains_predictor_without_moving_departure_chooser(monkeypatch)''', '''def test_residual_credit_replays_current_chooser_and_has_its_own_baseline(monkeypatch):
    """Held forecasts credit the shared chooser without an EMA or replay."""
    from test_item6_2_thinking import test_expectation_credit_moves_shared_chooser_toward_the_completed_chain
    test_expectation_credit_moves_shared_chooser_toward_the_completed_chain(monkeypatch)''')
name='test/test_normal_thought_controller.py';s=files[name]
for case in ['test_normal_boundary_uses_selected_semantic_meaning_as_its_answer_seed','test_normal_boundary_realizes_the_controller_selected_operation']:
    start=s.index('def '+case+'(');end=s.find('\ndef ',start+1)
    part=s[start:end].replace('registry.form("equal", _part, _part)', 'registry.form("equal", _leaves[0], _leaves[0])')
    s=s[:start]+part+s[end:]
files[name]=s
undo('test/test_operators_round2d.py', '''active=torch.ones(count,dtype=torch.bool),
                   compose_round=torch.arange(count).remainder(3))''','active=torch.ones(count,dtype=torch.bool))')
undo('test/test_operators_round2d.py', '''sentence_ids=torch.tensor([[0,1]]).expand(4,-1),sentence=0,
        compose_round=torch.tensor([-1,1,-1,1]))''', 'sentence_ids=torch.tensor([[0,1]]).expand(4,-1),sentence=0)')
undo('test/test_operators_round2d.py', 'active=torch.tensor([True,True]), compose_round=torch.tensor([2,-1]))', 'active=torch.tensor([True,True]))')
undo('test/test_thought_review.py', 'score=lambda result: dict(reconstruction=float(bool(open_slots(result.meaning))), answer=0.))', 'score=lambda result: float(bool(open_slots(result.meaning))))')
wrong=[name for name,data in files.items() if hashlib.sha256(data.encode()).hexdigest()!=expected[name]]
assert not wrong,wrong
with zipfile.ZipFile(HERE/'development-epoch-02-source.zip','x',zipfile.ZIP_DEFLATED) as archive:
    for name,data in files.items():archive.writestr(name,data)
print('Archived and verified',len(files),'source files against the dry-run manifest.')
