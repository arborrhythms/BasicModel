"""The fixed, teacher-only acquisition stage of item 1.8 §38.3."""
import json
from contextlib import contextmanager
from pathlib import Path

import torch


class CurriculumBoundReached(AssertionError):
    def __init__(self, report):
        self.report = report
        super().__init__('grammar curriculum reached its 200-round bound: ' + json.dumps(report))


def configure(model, examples, question):
    from util import TheXMLConfig
    assert len(examples) == 3
    data = model.inputSpace.data
    data.has_supervised_outputs = True
    data.text_answers = {'train': [item['text'] for item in examples]}
    annotation = json.loads(Path('data/scalar_lessons.json').read_text())['recognition_grammar']
    data.grammar_lessons = {'train': [dict(annotation, generation=dict(
        annotation['generation'], text=item['text'])) for item in examples]}
    data.train_input = [question] * len(examples)
    data.train_output = [torch.zeros(1) for _ in examples]
    data.source_addresses['train'] = [dict(split='train', document=i,
        external_id=f'scalar-question-{i}', sentence=0) for i in range(len(examples))]
    return model.getOptimizer(lr=float(TheXMLConfig.get('architecture.training.learningRate')))


def train_round(model, examples, question, optimizer):
    from What import What
    for index, item in enumerate(examples):
        model.train()
        inputs = model.inputSpace.prepInput([dict(text=question, scalar=item['scalar'])])
        model.runBatch(train=True, batchNum=index, batchSize=1, split='train',
            optimizer=optimizer, batch_override=(inputs, torch.zeros(1, 1)),
            source_rows=[index], questions=(What.supervised(index, prompt=question),))
        model.End()
    model.eval()


from Assessment import reading_forest


@contextmanager
def taught_probabilities(model):
    """Observe the existing teacher's inputs without rescoring or random draws."""
    from types import SimpleNamespace
    from GrammarLessons import constituent_examples, student_target_actions
    original = model._grammar_lesson_objectives
    calls = []
    def observe(programs, **kwargs):
        result = original(programs, **kwargs)
        table = model.inputSpace.data.grammar_lessons['train']
        for index, program in enumerate(programs):
            states = kwargs['student_states'][index]
            if states is None:
                continue
            view = SimpleNamespace(leaves=program.leaves, student_states=states)
            examples = constituent_examples(model.languageSpace, view,
                table[kwargs['source_rows'][index]]['tree'])
            calls.append([dict(spans=state['spans'], target=state['target'],
                probability=float(state['observed_probabilities'][student_target_actions(
                    model.languageSpace._tree_layer(2), state)].sum())) for state in examples])
        return result
    model._grammar_lesson_objectives = observe
    try:
        yield calls
    finally:
        model._grammar_lesson_objectives = original


def read_round(model, examples, question):
    model.eval()
    readings = []
    original = model._sentence_observation
    captured = []
    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get('admit'):
            # Admission observes the winner after its scored record has left
            # this call. Reconstruct its read-only word-position view from the
            # same kept state; no compose action is run or supplied here.
            record = model._trial_understanding(*args[:3])
            captured[:] = [reading_forest(model.languageSpace, program,
                model._sentence_leaf_positions(record, row).tolist())
                for row, program in enumerate(result['entries'])]
        return result
    model._sentence_observation = observe
    try:
        with torch.no_grad():
            for item in examples:
                captured.clear()
                model.understand(model.inputSpace.prepInput([dict(text=question, scalar=item['scalar'])]))
                readings.append(captured[0] if captured else [])
                model.End()
    finally:
        model._sentence_observation = original
    return readings


def bounded_stage(train, read, target, *, report=None):
    """Three consecutive complete rounds, or the specified hard bound."""
    consecutive = 0
    for round_index in range(1, 201):
        probabilities = train()
        readings = read()
        good = len(readings) == 3 and all(forest == [target] for forest in readings)
        consecutive = consecutive + 1 if good else 0
        result = dict(round=round_index, good=good, consecutive=consecutive,
                      readings=readings, taught_probabilities=probabilities)
        if report is not None:
            report(result)
        if consecutive == 3:
            return result
    raise CurriculumBoundReached(result)


def acquire(model, examples, question):
    optimizer = configure(model, examples, question)
    target = model.inputSpace.data.grammar_lessons['train'][0]['tree']
    with taught_probabilities(model) as calls:
        def train():
            calls.clear()
            train_round(model, examples, question, optimizer)
            return list(calls)
        return bounded_stage(train, lambda: read_round(model, examples, question), target,
            report=lambda result: print('GRAMMAR_CURRICULUM ' + json.dumps(result), flush=True))
