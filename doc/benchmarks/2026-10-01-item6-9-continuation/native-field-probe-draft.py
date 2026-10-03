"""Append these focused cases after the in-flight source finishes."""
def test_native_trial_reads_the_completed_point_with_live_credit():
    from Meaning import ConceptualMeaning
    from Models import BasicModel
    from What import What
    model, state, parameters = _sentence(True)
    root = state[1][9][:, 0]
    meanings = [ConceptualMeaning(torch.stack((row, row, torch.zeros_like(row))),
                                  torch.tensor([True, True, False])) for row in root]
    clauses = [SimpleNamespace(point=row, relation=None, meaning=meaning)
               for row, meaning in zip(root, meanings)]
    model.answer_synthesis = True
    model._sentence_answer_questions = tuple(What.supervised(i) for i in range(2))
    model.inputSpace.data.what = lambda question: SimpleNamespace(available=True, what=0.)
    model._what_grammar_context = lambda *args, **kwargs: (None, None)
    model.outputSpace.prepOutput = lambda values: torch.tensor(values)[:, None]
    seen = []
    def realize(understanding, derivation, *, detach_understanding=True):
        assert not detach_understanding
        assert [int(field.meaning.role_mask.sum()) for field in derivation.sentence_states] == [1, 1]
        seen.append(derivation.conceptual_answer)
        return SimpleNamespace(actual=(derivation.conceptual_answer.flatten(1) @ parameters[-1])[:, None])
    model.reverseOutput = realize
    error = BasicModel._sentence_answer_error(model, state, 0, torch.ones(2, dtype=torch.bool),
                                              dict(meanings=meanings, clauses=clauses))
    assert len(seen) == 1
    torch.testing.assert_close(seen[0][:, 0], root)
    gradients = torch.autograd.grad(error.sum(), parameters, allow_unused=True)
    assert all(g is not None and bool(g.abs().any()) for g in gradients)


def test_automatic_answers_do_not_start_a_supplied_answer_trial():
    from Models import BasicModel
    from What import What
    model, state, _ = _sentence(True)
    model.answer_synthesis = True
    model._sentence_answer_questions = tuple(What.future(i) for i in range(2))
    model.inputSpace.data.what = lambda question: SimpleNamespace(available=True, what='automatic')
    def forbidden(*args, **kwargs):
        raise AssertionError('an automatic answer must not start a supervised generation trial')
    model.reverseOutput = forbidden
    model._what_grammar_context = lambda *args, **kwargs: (None, None)
    assert BasicModel._sentence_answer_error(model, state, 0, torch.ones(2, dtype=torch.bool),
                                              dict(meanings=[None, None])) is None
