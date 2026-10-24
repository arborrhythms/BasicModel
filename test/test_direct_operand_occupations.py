"""Repeated lexical occurrences flow through direct operands and end at a clause."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from PartsOfSpeech import PartsOfSpeech, formal_arguments, flow_mask, operation_name, update_flow


@pytest.mark.parametrize('compiled', (False, True))
def test_repeated_occurrences_keep_direct_counts_through_unary_and_clause(compiled):
    from Language import Grammar
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    rules = {operation_name(rule): rule for rule in grammar.rules_upward if operation_name(rule)}
    binary = tuple(rules[name] for name in ('intersection', 'compound', 'lift'))
    unary = (rules['compound_unary'],)
    parts = PartsOfSpeech(formal_arguments((*binary, *unary)))
    # Arbitrary, distinct prior distributions expose a wrong occurrence gather.
    # The same word occurs twice; its second occurrence is not another word type.
    for word, argument in zip((101, 102, 103), parts.arguments):
        parts.add(word, argument)
    initial = {word: counts.copy() for word, counts in parts.counts.items()}
    word_ids = torch.tensor([101, 102, 102, 103])
    actions = ((0, -1, 0), (0, -1, 1), (0, -1, 2),
               (1, 0, 0), (1, 1, 0), (2, 0, 0),
               (0, -1, 3), (1, 2, 0), (2, 0, 0))
    program = SimpleNamespace(word_ids=word_ids, actions=torch.tensor(actions))
    probabilities = parts.probabilities(word_ids, torch.zeros(1))
    binary_flow, unary_flow = tuple(map(flow_mask, binary)), tuple(map(flow_mask, unary))

    choices = tuple(None if kind == 0 else SimpleNamespace(
        applied=torch.tensor([True]), kind=torch.tensor([kind]),
        local_op=torch.tensor([local]), position=torch.tensor([position]))
        for kind, local, position in actions)

    def carry(probabilities):
        slab = probabilities.new_zeros(1, 5, probabilities.shape[-1])
        history = []
        for (kind, local, position), choice in zip(actions, choices):
            if kind == 0:
                slab = torch.cat((probabilities[position][None, None], slab[:, :-1]), 1)
            else:
                slab = update_flow(slab, choice, binary_flow, unary_flow)
            history.append(slab)
        return torch.stack(history)

    run = torch.compile(carry, backend='aot_eager', fullgraph=True) if compiled else carry
    observed = run(probabilities)
    a, b, _, c = probabilities
    # Independently stated word-occurrence carriers after each grammar step.
    expected_words = ((a,), (b, a), (b, b, a), (2*b, a), (2*b,),
                      (2*b,), (c, 2*b), (), ())
    expected = torch.zeros_like(observed)
    for index, words in enumerate(expected_words):
        if words:
            expected[index, 0, :len(words)] = torch.stack(words)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert parts.counts == initial

    parts.observe_kept(program, binary, unary)
    delta = {word: parts.counts[word] - initial[word] for word in initial}
    assert delta[101] == Counter(compound_argin1=1)
    assert delta[102] == Counter(intersection_argin1=1, intersection_argin2=1,
        intersection_argout1=2, compound_argin2=2, compound_argout1=2,
        compound_unary_argin1=2, compound_unary_argout1=2, lift_argin1=2)
    assert delta[103] == Counter(lift_argin2=1)
    # Clause completion clears both occurrences before the final unary call.
    assert observed[-2:].count_nonzero() == 0
