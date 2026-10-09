"""Diagnostic saved-entropy replay, separate from all declared attempts."""
import hashlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent


def main(source, variant, start):
    sys.path[:0] = [str(source/'bin'), str(source/'test')]
    import numpy as np
    import torch
    from test_mm_xor import _fresh_model
    from util import init_device
    init_device('cpu')
    torch.set_num_threads(1)
    saved = torch.load(HERE/'measurements'/start/'unseeded-entry.pt',
                       weights_only=False, map_location='cpu')
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch'])

    def rng():
        return dict(torch=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
            numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest())

    model, _, data = _fresh_model(str(source/'data/MM_xor.xml'))
    def parameters():
        return {name:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for name,value in model.named_parameters()}
    result = dict(kind='diagnostic replay only', start=start, variant=variant,
        seed=None, construction_rng=rng(), initial_parameters=parameters(), trajectory=[])
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    criterion = torch.nn.MSELoss()
    loader = data.data_loader(split='train',num_streams=4)
    try:
        for epoch in range(1,201):
            raw, target = next(iter(loader))
            inp = model.inputSpace.prepInput(raw)
            expected = model.outputSpace.prepOutput(target)
            before = rng()
            optimizer.zero_grad()
            _, _, output, _ = model.forward(inp)
            after = rng()
            expected = expected.to(output.device)
            while expected.dim() < output.dim():
                expected = expected.unsqueeze(-1)
            loss = criterion(output, expected.expand_as(output))
            loss.backward()
            optimizer.step()
            row = dict(epoch=epoch,mse=float(loss.detach()),predictions=output.detach().reshape(-1).tolist(),
                before_rng=before,after_rng=after,
                ir_mask=model._ir_mask_positions.detach().tolist(),
                narrowing=model._attention_words.actions.detach().tolist())
            result['trajectory'].append(row)
            if row['mse'] < .20:
                break
        result.update(best=min(row['mse'] for row in result['trajectory']),
            steps=len(result['trajectory']), final_parameters=parameters(), final_rng=rng())
        path = HERE/'mm-bisection'/f'{start}-{variant}.json'
        with path.open('x') as stream:
            json.dump(result,stream,indent=2)
            stream.write('\n')
        print(json.dumps(dict(start=start,variant=variant,steps=result['steps'],best=result['best'])))
    finally:
        model.End()
        model.symbolSpace.soft_reset()


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(),sys.argv[2],sys.argv[3])
