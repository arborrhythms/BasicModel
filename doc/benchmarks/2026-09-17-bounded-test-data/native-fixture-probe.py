"""Narrow development fixtures still cross distinct conceptual/percept widths."""
import torch
from test_output_path_supervised import _native_answer_model
from test_output_walk import _capture_program_probe
from What import What


def test_development_native_fixture_preserves_wide_concept_gradient(tmp_path):
    model=_native_answer_model(tmp_path,False,concept_width=264)
    try:
        with torch.no_grad():
            understood=_capture_program_probe(model,['1 plus 2','3 plus 4'])
            answer=model.resolveAnswer(understood,(What.supervised(0),What.supervised(1)))
        from dataclasses import replace
        source=answer.conceptual_answer.detach().clone().requires_grad_()
        held=replace(answer,conceptual_answer=source)
        result=model.reverseOutput(understood,held)
        gradient=torch.autograd.grad(result.percepts.square().mean(),source)[0]
        assert source.shape[-1]==264 and result.percepts.shape[-1]==136
        assert torch.isfinite(gradient).all()
        assert gradient[...,136:].abs().sum()>0
    finally:
        model.End();model.symbolSpace.soft_reset();torch._dynamo.reset()
