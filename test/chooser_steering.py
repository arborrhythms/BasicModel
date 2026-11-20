"""A stand-in scorer for tests of the operation layer's mechanics.

The chooser of acts is a network (``MLPTransformChooser``); a test of the
layer's menu, softmax, deadlines or credit needs a choice it can name in
advance. ``stand_in`` puts an inner-product scorer in the chooser's place for
that test: each act's score is the candidate's inner product with a vector of
the act (``stop`` for STOP/wait, ``reduce[i]``, ``unary[j]``, ``bracket[k]``),
which the test sets. The vectors are parameters, so a test can see the
score-function credit reach the scorer. Nothing in ``bin/`` reads them.
"""
import types

import torch
from torch import nn


def stand_in(layer):
    """Score ``layer``'s acts by inner products with the returned vectors."""
    d = int(layer.d_model)
    scorer = nn.Module()
    scorer.stop = nn.Parameter(torch.randn(1, d) * .02)
    scorer.reduce = nn.Parameter(torch.randn(int(layer.r_reduce), d) * .02)
    scorer.unary = nn.Parameter(torch.randn(int(layer.r_apply), d) * .02)
    scorer.bracket = nn.Parameter(torch.zeros(6, d))
    chooser = layer.chooser

    def rows():
        return torch.cat((scorer.reduce, scorer.unary, scorer.bracket))

    def score_unary(self, x_score, applied_score, *, what_ctx=None, op_offset=0,
                    op_indices=None, reference_ctx=None, budget=None):
        copy = torch.einsum('bnd,od->bno', x_score, scorer.stop.to(x_score))
        count = applied_score.shape[2] if applied_score.dim() == 4 else 0
        if not count:
            return copy, x_score.new_zeros(*x_score.shape[:2], 0)
        index = torch.arange(count, device=x_score.device) if op_indices is None else op_indices
        anchors = rows().to(x_score).index_select(0, index + op_offset)
        return copy, torch.einsum('bnad,ad->bna', applied_score, anchors)

    def score_binary(self, x_score, reduced_score, *, what_ctx=None, op_indices=None,
                     reference_ctx=None, budget=None):
        copy = torch.einsum('bnd,od->bno', x_score, scorer.stop.to(x_score))
        if reduced_score.dim() != 4 or not reduced_score.shape[1] or not reduced_score.shape[2]:
            count = int(reduced_score.shape[2]) if reduced_score.dim() == 4 else int(layer.r_reduce)
            return copy, x_score.new_zeros(x_score.shape[0], max(x_score.shape[1] - 1, 0), count)
        count = int(reduced_score.shape[2])
        index = torch.arange(count, device=x_score.device) if op_indices is None else op_indices
        anchors = rows().to(x_score).index_select(0, index)
        return copy, torch.einsum('bnrd,rd->bnr', reduced_score, anchors)

    chooser.score_unary = types.MethodType(score_unary, chooser)
    chooser.score_binary = types.MethodType(score_binary, chooser)
    layer.stand_in = scorer
    return scorer
