"""Isolate native promotion at one parameter state; no optimizer or seed."""
import json
from pathlib import Path
import warnings

from probe_mm_xor import ROOT, _fresh_model, geometry, init_device, tensor, torch


def run():
    init_device('cpu')
    torch.set_num_threads(1)
    model, _, _ = _fresh_model(str(ROOT / 'data/MM_xor.xml'))
    captured = {}
    original = model.outputSpace.forwardLinear

    def readout(value):
        captured['readout_input'] = value
        return original(value)

    model.outputSpace.forwardLinear = readout
    stages = []
    for index, stage in enumerate(model.body_stages):
        combine = getattr(stage['cs'], 'combine', None)
        stages.append(dict(stage=index, merge_present='merge' in stage,
            combine=type(combine).__name__ if combine is not None else None,
            nonlinear=getattr(combine, 'nonlinear', None),
            mode=getattr(combine, 'sigma_pi_mode', None)))
    report = dict(seed=None, optimizer_steps=0, serial=model.serial,
                  configured_grammar=model.useGrammar, stages=stages, passes=[])
    loader = model.inputSpace.data.data_loader(split='train', num_streams=4)
    try:
        for turn in range(3):
            texts, answers = next(iter(loader))
            inp = model.inputSpace.prepInput(texts)
            target = model.outputSpace.prepOutput(answers)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _, _, prediction, _ = model.forward(inp)
            target = target.to(prediction).reshape_as(prediction)
            loss = (prediction-target).square().mean()
            x = captured['readout_input']
            actual_weight = torch.autograd.grad(prediction.sum(), x, retain_graph=True)[0]
            state = model.perceptualSpace._forward_input
            model.zero_grad(set_to_none=True)
            loss.backward()
            grad_summary = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    grad_summary[name] = dict(norm=float(parameter.grad.norm()),
                                              nonzero=int(torch.count_nonzero(parameter.grad)))
            store = model.perceptualSpace.percept_store
            words = {}
            for word in ('hello', 'loving', 'world', 'there', ' '):
                row = store.get_id(word.encode('utf-8'))
                words[word] = None if row is None else dict(row=int(row),
                    master=tensor(store._basis.W[row]),
                    content=tensor(store._basis.lookup_rows(row)),
                    gradient=tensor(store._basis.W.grad[row]) if store._basis.W.grad is not None else None)
            report['passes'].append(dict(turn=turn, loss=float(loss.detach()),
                texts=texts, word_texts=state['word_texts'],
                ids=tensor(state['indices']), groups=tensor(state['word_groups']),
                embedded=tensor(model.perceptualSpace._embedded_input), words=words,
                readout_input=geometry(x, target), actual_readout_weight_norm=float(actual_weight[0].norm()),
                derivations=None,
                derivation_reason='Parallel reading: _chart_compose_at_C returns before routing; no merge modules installed.',
                gradients=grad_summary))
        Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(stages=stages, passes=[dict(turn=p['turn'],
            loss=p['loss'], singular_values=p['readout_input']['centered_singular_values'],
            affine_weight_norm=p['readout_input']['affine_weight_norm'],
            actual_weight_norm=p['actual_readout_weight_norm']) for p in report['passes']])), flush=True)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


if __name__ == '__main__':
    run()
