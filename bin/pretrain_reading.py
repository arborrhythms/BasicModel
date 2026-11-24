"""Teach free occurrence selection, then certify the saved reader on unseen sequences."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from What import What

from Assessment import assess_age, preserved
from util import parse

ROOT = Path(__file__).resolve().parents[1]
LESSON = ROOT / 'data/lessons/reading-order.json'


def load_lesson(path):
    spec = json.loads(Path(path).read_text())
    seen = set()
    for split in ('train', 'validation', 'test'):
        texts = spec[split]
        if not texts or len(set(texts)) != len(texts) or seen.intersection(texts):
            raise ValueError(f'{split} must contain distinct, held-out sequences')
        seen.update(texts)
    if not set(spec['required']).issubset(spec['test']):
        raise ValueError('the required regression pairs must be in the test split')
    if not 0 < spec['minimum_accuracy'] <= 1 or min(spec['epochs'], spec['batch_size']) < 1:
        raise ValueError('invalid reading lesson limits')
    return spec


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def staged_input(model, texts, size):
    # The native compiled bricks retain the same stream width through a lesson.
    padded = list(texts) + [''] * (size - len(texts))
    return model.inputSpace.prepInput(padded)


def words(text):
    return [form for form, _ in parse(text, lex='words') if form.strip()]


@torch.no_grad()
def presentation(model, texts, batch_size):
    """Measure the completed walk; source order is visible only to the assessor."""
    try:
        model.understand(staged_input(model, texts, batch_size))
        reading = model._attention_words
        if reading is None:
            raise ValueError('reading pretraining requires a model with an occurrence walk')
        forms = model._attention_forms[0]
        actions = reading.actions.tolist()
        accepted = reading.accepted.tolist()
        correct = []
        for b, text in enumerate(texts):
            slots = [i for i, form in enumerate(forms[b]) if form]
            if [forms[b][i] for i in slots] != words(text):
                raise ValueError(f'input does not fit the model or lesson tokenizer: {text!r}')
            taken = [index for index in actions[b] if index >= 0]
            correct.append(taken == slots and all(accepted[b][i] for i in slots))
        return correct
    finally:
        model.End()
        model.dispatch_per_row_reset([True] * batch_size)


def assessment(model, texts, batch_size, *, sample):
    failures = []
    from eval_nanochat_grammar import frozen_online_learning
    with preserved(model), frozen_online_learning(model):
        model.train(sample)
        for batch in batches(texts, batch_size):
            result = presentation(model, batch, batch_size)
            failures.extend(text for text, ok in zip(batch, result, strict=True) if not ok)
    return dict(correct=len(texts) - len(failures), total=len(texts), failures=failures)


def qualifies(result, required, threshold):
    if result.keys() != {'sampled', 'evaluation'}:
        return False
    return all(row['total'] and row['correct'] / row['total'] >= threshold
               and not set(required).intersection(row['failures']) for row in result.values())


def measure(model, texts, size):
    return {name: assessment(model, texts, size, sample=sample)
            for name, sample in [('sampled', True), ('evaluation', False)]}


def train_reader(model, spec, optimizer):
    size = spec['batch_size']
    # All vocabulary admission uses ordinary presentations of training words.
    vocabulary = sorted({w for t in spec['train'] for w in words(t)})
    with torch.no_grad():
        model.eval()
        for _ in range(model._concept_owner().concept_mint_count):
            for batch in batches(vocabulary, size):
                model.understand(staged_input(model, batch, size))
                model.End()
                model.dispatch_per_row_reset([True] * size)
    history = []
    for epoch in range(1, spec['epochs'] + 1):
        model.train()
        train = [spec['train'][i] for i in torch.randperm(len(spec['train'])).tolist()]
        for batch in batches(train, size):
            # Keep this unlabeled presentation independent of the corpus
            # whose grammar will run after the prerequisite.
            data = model.inputSpace.data
            supervised = data.has_supervised_outputs
            try:
                with data.runtime_batch(batch):
                    data.has_supervised_outputs = False
                    source = model.inputSpace.prepInput(batch)
                    questions = tuple(What.present(i, split='runtime') for i in range(len(batch)))
                    model.runBatch(train=True, optimizer=optimizer, batchSize=len(batch),
                        split='runtime', questions=questions,
                        batch_override=(source, torch.empty(len(batch), 0, device=source.device)))
            finally:
                data.has_supervised_outputs = supervised
                model.End()
                model.dispatch_per_row_reset([True] * len(batch))
        results = measure(model, spec['validation'], size)
        history.append(results)
        print('epoch', epoch, {name: (r['correct'], r['total']) for name, r in results.items()}, flush=True)
        if qualifies(results, (), spec['minimum_accuracy']):
            break
    return history


def run(config, lesson, output):
    from Models import BaseModel
    from data import Data
    spec = load_lesson(lesson)
    output.mkdir(parents=True, exist_ok=False)
    data = Data()
    model, _ = BaseModel.from_config(str(config), data=data)
    manifest = dict(dataset='reading-order', training='sentence-reconstruction-and-expectation', sha256=hashlib.sha256(lesson.read_bytes()).hexdigest())
    model.inputSpace.data.source_manifest = manifest
    report = dict(lesson=manifest, config=str(config.resolve()),
                  config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), epochs=[])
    size = spec['batch_size']
    try:
        optimizer = model.getOptimizer()
        model._optimizer = optimizer
        report['epochs'] = train_reader(model, spec, optimizer)
        report['age'] = assess_age(model, all_rows=False)['age']
        pending = output / 'candidate.ckpt'
        model.save_weights(str(pending))
    finally:
        model.End()
    restored, _ = BaseModel.from_config(str(config), data=data)
    try:
        if not restored.load_weights(str(pending), require_match=True):
            raise ValueError('could not restore reader')
        if float(restored.age) != float(restored.age.new_tensor(report['age'])):
            raise ValueError('saved reader lost its assessed age')
        results = measure(restored, spec['test'], size)
        report['test'] = results
        report['passed'] = (qualifies(report['epochs'][-1], (), spec['minimum_accuracy'])
                            and qualifies(results, spec['required'], spec['minimum_accuracy']))
        with pending.open('rb') as stream:
            report['checkpoint_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if report['passed']:
            pending.rename(output / 'reader.ckpt')
        (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        print('held out', {name: (r['correct'], r['total']) for name, r in results.items()},
              'age', report['age'], 'passed', report['passed'], flush=True)
        return report
    finally:
        restored.End()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'data/XOR_grammar.xml')
    parser.add_argument('--lesson', type=Path, default=LESSON)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raise SystemExit(0 if run(args.config, args.lesson, args.output)['passed'] else 1)
