import json
import ClauseJournal
from test_space_equiv_selfcheck import test_identity_candidate_passes as run_original


def test_identity_candidate_closing_diagnostic(monkeypatch):
    original = ClauseJournal.finish_clause
    def observe(language, program, **kwargs):
        try:
            return original(language, program, **kwargs)
        except ValueError:
            print(json.dumps(dict(depth=kwargs.get('depth'), actions=program.actions.tolist(),
                end_state=list(program.end_state.shape), leaves=list(program.leaves.shape),
                binary=[r.method_name for r in language._compose_binary_rules],
                unary=[r.method_name for r in language._compose_unary_rules])))
            raise
    monkeypatch.setattr(ClauseJournal, 'finish_clause', observe)
    run_original()
