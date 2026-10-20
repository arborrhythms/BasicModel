"""External configuration/observation harness for the two standing-gate arms.

The XML parser overlay reaches the native CLI too. No assertion, optimizer,
training budget, selection action or result is replaced. Tensor observations
are omitted inside compiled graphs, so coverage counts describe eager reads.
"""
import atexit
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ARM = os.environ.get('ITEM4_GATE_ARM')
ROOT = Path(__file__).resolve().parents[3]
RECEIPT = Path(__file__).resolve().parent
STATE = {}
SEEN = set()
MODEL_COUNTERS = {}
PATCHED = set()
INSTALLED = False


def case():
    return os.environ.get('ITEM4_GATE_CASE', 'collection')


def record():
    return STATE.setdefault(case(), dict(seed=os.environ.get('ITEM4_GATE_SEED'),
        configuration={}, constructors=[], reads={}))


def token():
    return hashlib.sha256(case().encode()).hexdigest()[:20]


def begin(group):
    import numpy as np
    import torch
    nonce = os.environ['ITEM4_GATE_NONCE']
    seed = int.from_bytes(hashlib.sha256((nonce + group).encode()).digest()[:4], 'big')
    os.environ['ITEM4_GATE_CASE'] = group
    os.environ['ITEM4_GATE_SEED'] = str(seed)
    os.environ['BASIC_SEED'] = str(seed)
    if group not in SEEN:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        SEEN.add(group)
    record()['seed'] = seed


def flush():
    if not STATE:
        return
    directory = RECEIPT / os.environ.get('ITEM4_GATE_METRICS', 'all-levels-metrics') / ARM
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'process-{os.getpid()}.json').write_text(json.dumps(STATE, indent=2) + '\n')


def gain_read(label, value, *, prior=True):
    import torch
    if torch.compiler.is_compiling() or not torch.is_tensor(value) or not value.numel():
        return
    v = value.detach()
    counts = record()['reads'].setdefault(label, dict(calls=0, with_prior=0,
        coordinates=0, below_one=0, minimum=1., maximum=0.))
    counts['calls'] += 1
    counts['with_prior'] += int(prior)
    counts['coordinates'] += v.numel()
    counts['below_one'] += int((v < 1.).sum())
    counts['minimum'] = min(counts['minimum'], float(v.min()))
    counts['maximum'] = max(counts['maximum'], float(v.max()))


def patch_model(module):
    import torch
    if module is None or not hasattr(module, 'BasicModel') or id(module.BasicModel) in PATCHED:
        return
    cls = module.BasicModel
    PATCHED.add(id(cls))
    original = cls._create_per_stage

    def create(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        index = MODEL_COUNTERS.get(case(), 0)
        MODEL_COUNTERS[case()] = index + 1
        # Temporary initial parameters, never a trained phase artifact.
        directory = ROOT / 'output/item4-step0-initial' / token()
        directory.mkdir(parents=True, exist_ok=True)
        # CLI process construction is distinct from its pytest parent.
        suffix = 'cli' if module.__name__ == '__main__' else 'test'
        path = directory / f'{suffix}-{index}.pt'
        params = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        item = dict(index=index, scope='common per-stage constructor', parameters=len(params),
            gains=[model.percept_expectation_gain, model.word_expectation_gain,
                   model.expectation_gain, model.document_expectation_gain],
            prime=model.expectation_prime_enabled)
        if ARM == 'zero':
            if path.exists():
                baseline = torch.load(path, weights_only=True, map_location='cpu')
                same = params.keys() == baseline.keys() and all(
                    torch.equal(params[name], baseline[name]) for name in params)
                item['initial_parameters_equal_to_recorded_run'] = same
                if not same:
                    raise AssertionError('scoped replay changed the recorded initialization')
            else:
                torch.save(params, path)
                item['initial_parameters'] = 'saved for paired comparison'
        else:
            baseline = torch.load(path, weights_only=True, map_location='cpu')
            unequal = sorted(set(params) ^ set(baseline))
            unequal += [name for name in params.keys() & baseline.keys()
                        if not torch.equal(params[name], baseline[name])]
            item.update(initial_parameters_equal=not unequal, unequal_parameters=unequal)
        record()['constructors'].append(item)
        flush()
        return result
    cls._create_per_stage = create


def install():
    global INSTALLED
    if INSTALLED or not ARM:
        return
    INSTALLED = True
    import torch
    import util
    configuration = dict(sentenceExpectation=True, expectationPrime=True,
        explorationUniformFraction=.5, **{name: 0. if ARM == 'zero' else .5 for name in (
            'perceptExpectationGain', 'symbolExpectationGain', 'rowExpectationGain',
            'documentExpectationGain')})
    parse = util.XMLConfig._parse_xml

    def overlay(path):
        data = parse(path)
        training = data.setdefault('architecture', {}).setdefault('training', {})
        training.update(configuration)
        if os.environ.get('ITEM4_GATE_SEED'):
            training['seed'] = int(os.environ['ITEM4_GATE_SEED'])
        record()['configuration'] = configuration
        patch_model(sys.modules.get('__main__'))
        return data
    util.XMLConfig._parse_xml = staticmethod(overlay)
    import Models
    patch_model(Models)

    from Expectation import PerceptMoments, SymbolRead
    region_gain, symbol_gain = PerceptMoments.region_gain, SymbolRead.gains
    def percept(owner, *args, **kwargs):
        result = region_gain(owner, *args, **kwargs)
        if not torch.compiler.is_compiling():
            gain_read('P', result, prior=bool(owner.count > 0))
        return result
    def symbol(owner, *args, **kwargs):
        result = symbol_gain(owner, *args, **kwargs)
        if not torch.compiler.is_compiling():
            gain_read('S', result)
        return result
    PerceptMoments.region_gain, SymbolRead.gains = percept, symbol

    import WordExpectation
    word_distribution = WordExpectation.word_distribution
    def word(*args, **kwargs):
        result = word_distribution(*args, **kwargs)
        if not torch.compiler.is_compiling():
            gain_read('S_word', result.gain)
        return result
    WordExpectation.word_distribution = word

    from Meaning import ClosingImage
    closing = ClosingImage.form.__func__
    def form(cls, *args, **kwargs):
        result = closing(cls, *args, **kwargs)
        if not torch.compiler.is_compiling():
            frame = sys._getframe(1)
            prediction = kwargs.get('prediction')
            row = prediction is not None and prediction is frame.f_locals.get('row_prior')
            label = 'R' if row else 'D' if frame.f_code.co_name == '_commit_sentence' else 'other_closing'
            gain_read(label, result.gain, prior=prediction is not None)
        return result
    ClosingImage.form = classmethod(form)
    atexit.register(flush)
