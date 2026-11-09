"""Formal argument occupation scores for parent and proposed children."""
import torch
from PartsOfSpeech import operation_name


def features(language, parent, left, right):
    result = parent.new_zeros(*left.shape[:-1], 3)
    speech = language.parts_of_speech
    rules = (*language._generate_binary_rules, *language._generate_unary_rules)
    parent_scores = speech.value_scores(parent)
    left_scores, right_scores = speech.value_scores(left), speech.value_scores(right)
    for index, rule in enumerate(rules):
        name = operation_name(rule)
        for lane, face, scores in ((0, 'argout1', parent_scores),
                                  (1, 'argin1', left_scores[:, index]),
                                  (2, 'argin2', right_scores[:, index])):
            argument = speech.index.get(name + '_' + face) if name else None
            if argument is not None:
                result[:, index, lane] = scores[..., argument]
    # No rule is attached to STOP. Undeclared fixed faces also remain zero.
    return result.detach()
