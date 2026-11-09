"""Every supplied training form has observable admission and terminal routing."""
import json
from pathlib import Path

import pytest
import torch


def model_for_admission(monkeypatch):
    import util
    from data import TheData
    from Models import BaseModel
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                      for split in ('train', 'validation', 'test')})
    return BaseModel.from_config('data/MM_corpus_lessons.xml', data=TheData)[0]


@pytest.mark.parametrize('answer', ['yes', 'omega'])
def test_supplied_form_is_reported_without_changing_retrieval(answer, monkeypatch):
    from Corpus import admit_training_forms
    from Generative import lexical_candidates
    from corpus_lessons import lexical_report
    model = model_for_admission(monkeypatch)
    try:
        owner = model._concept_owner()
        admit_training_forms(model, ['alpha.'], [answer])
        word = owner.interpret.word_for_form(answer)
        assert word is not None and not owner.interpret.denotations(word)
        row = owner._csw_row_of(owner.interpret.word_for_form('alpha'))
        root = owner.similarity_codebook.lookup_rows(torch.tensor([[row]]), streams=True)[:, 0]
        reports = []
        model._lexical_admission_sink = reports.append
        bank, report = lexical_candidates(owner, root, limit=16)
        form = report['supplied_forms'][0]
        assert form == dict(form=answer, admitted=True, shortlisted=[True],
                            exact_route_available=[True])
        assert reports == [report]
        model._last_lexical_retrieval = report
        assert lexical_report(model, ['alpha'])['forms'][answer]['admission'] == form
        model._supplied_training_forms = ()
        without_report, _ = lexical_candidates(owner, root, limit=16)
        torch.testing.assert_close(bank.rows, without_report.rows, rtol=0, atol=0)
        torch.testing.assert_close(bank.codes, without_report.codes, rtol=0, atol=0)
    finally:
        model.End()


def test_lesson_fixture_supplied_forms_reach_every_applicable_bank_without_training(monkeypatch):
    from Corpus import admit_training_forms
    from corpus_lessons import stage_documents
    from Generative import lexical_candidates
    from Meronomy import percept_spans
    from WordSurfaces import is_lexical_form
    from data import TheData
    model = model_for_admission(monkeypatch)
    try:
        owner = model._concept_owner()
        banks, answer_banks = 0, 0
        for stage in json.loads(Path('data/corpus-lessons.json').read_text())['stages']:
            texts, targets = stage_documents(TheData, stage)
            admit_training_forms(model, texts, targets)
            for text, target in zip(texts, targets):
                if not isinstance(target, str):
                    continue
                raw = text.encode('ascii')
                words = [raw[a:b] for a, b in percept_spans(raw) if is_lexical_form(raw[a:b])]
                rows = [owner._csw_row_of(owner.interpret.word_for_form(word)) for word in words]
                root = owner.similarity_codebook.lookup_rows(torch.tensor([rows]), streams=True).mean(1)
                _, report = lexical_candidates(owner, root, limit=model.reconstruction_basis_limit)
                supplied = {value['form']: value for value in report['supplied_forms']}
                for start, end in percept_spans(target.encode('ascii')):
                    form = target[start:end]
                    if not is_lexical_form(form.encode('ascii')):
                        continue
                    row = supplied[form]
                    assert row['admitted'] and all(row['shortlisted']) and all(row['exact_route_available'])
                    banks += 1
                    answer_banks += form == 'yes'
        assert banks == 20 and answer_banks == 8
    finally:
        model.End()
