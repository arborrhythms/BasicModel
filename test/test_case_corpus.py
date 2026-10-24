"""The phase corpus changes surfaces only, with evaluator-owned labels."""
from constituent_corpus import CASE_VARIANTS, case_corpus


def test_six_placements_share_a_factorial_and_keep_grading_separate():
    assert len(CASE_VARIANTS) == 6
    for variant in CASE_VARIANTS:
        rows = case_corpus(variant)
        assert len(rows) == 24
        assert sum(row['heldout'] for row in rows) == 6
        assert all(len(row['sentences']) == 2 and row['prompt'].endswith('?') for row in rows)
        assert all(row['answer'] not in ' '.join(row['sentences']).split() for row in rows)
        assert all(set(row['labels']) == {'case', 'head', 'selections', 'nested'} for row in rows)
        train = {text for row in rows if not row['heldout'] for text in row['sentences']}
        held = {text for row in rows if row['heldout'] for text in row['sentences']}
        assert any(text not in train for text in held)
    unmarked = case_corpus('unmarked')
    assert {row['answer'] for row in unmarked} == {'un'}
    assert all(row['labels']['selections'] == 0 for row in unmarked)
    for variant in ('one_leaf_two_selections', 'two_leaves'):
        assert all(row['labels']['selections'] == 2 for row in case_corpus(variant))
