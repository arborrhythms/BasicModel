"""Production capture must not label a type-valued fusion with another identity."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
from types import SimpleNamespace
import torch
from test_item7_references import language_and_program, frame, prediction
from reading_fixtures import record_reading
from Models import BasicModel
from ClauseJournal import finish_clause

language, initial = language_and_program()
reading = record_reading(language, initial)
prior = frame()
owner = SimpleNamespace(_concept_source_order=lambda cid: 1)
language.resolve_lexical_references = lambda *a, **k: (
    initial.concept_ids.clone(), torch.tensor([1, 1]))
trace = SimpleNamespace(_choice_values=reading.operation_values.reshape(1,3,-1),
                        _choice_refs=reading.operation_refs[None])
discourse=SimpleNamespace(situation_references=lambda row:(prior,),
    _inter_last_meaning=[SimpleNamespace(prediction=prediction(prior.point))])
model=SimpleNamespace(languageSpace=language,_concept_owner=lambda:owner,
    _reconstruction_stack=lambda:trace,symbolSpace=SimpleNamespace(discourse=discourse))
program=(torch.tensor([[0,1]]),reading.actions[None],reading.targets[None],torch.tensor([[0,1,2]]))
entry=BasicModel._program_entries(model,program,reading.leaves[None],reading.rows[None],
    reading.word_rows[None],reading.activations[None],reading.end_state[None],
    concept_ids=reading.concept_ids[None],admit=False)[0]
clause=finish_clause(language,entry)
print('selected identity',entry.reference_ids.tolist(), 'cached references',clause.refs)
print('actual point',clause.point.tolist(),'required point',(prior.point+initial.leaves[1]).tolist())
assert clause.refs[0] == prior.row_id
assert torch.equal(clause.point, prior.point+initial.leaves[1])
