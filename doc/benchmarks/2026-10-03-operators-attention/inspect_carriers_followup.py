import torch

def test_mixed_gradient_details(monkeypatch):
    import test_output_path_supervised as tests
    original=tests._answer_training_probe
    def run(m,*args):
        result=original(m,*args)
        c=m._last_answer_construction
        print('TRACE',m._last_output_walk_trace,'TRUNC',m._output_truncated,flush=True)
        print('CONCEPT_NORMS',c.concepts.norm(dim=-1),'IDEA',c.derivation.conceptual_answer.norm(dim=-1),'CONTEXT',c.derivation.conditioning_context,flush=True)
        print('GRADS',[None if g is None else float(g.norm()) for g in result['total_grads']],flush=True)
        return result
    monkeypatch.setattr(tests,'_answer_training_probe',run)
    tests.test_mixed_supplied_numeric_and_automatic_text_trains_only_supplied_row(None)
