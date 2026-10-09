"""Observe completed native thoughts; exact checks are verifier-side only."""
from collections import Counter
from contextlib import ExitStack
import json
from unittest.mock import patch

import torch


def encode(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    return value


class Observer:
    def __init__(self, folder):
        self.folder = folder
        self.counts = Counter()
        self.operations = Counter()
        self.credits = Counter()
        self.results = {}
        self.source_texts = {}
        self.questions = []
        self.context = {}
        self.credit_sample = []
        self.gradient_l2_sum = 0.
        self.gradient_l2_max = 0.
        self.question_log = (folder / 'questions.jsonl').open('w')

    def __enter__(self):
        from Models import BasicModel
        from ThoughtReferences import open_slots
        import ThoughtCredit
        import EqualityLearning
        self.stack = ExitStack()
        run = BasicModel.run_selected_thought
        register = ThoughtCredit.register
        equality_cost = EqualityLearning.cost

        def equality(language, programs, active, like):
            value, counts = equality_cost(language, programs, active, like)
            acts = sum(counts)
            self.counts['equality_loss_trial_acts'] += acts
            if acts and value.requires_grad:
                parameters = tuple(parameter for name, parameter in
                    language.language_layer.operation_layer.named_parameters()
                    if name.endswith('_verb_shift') and parameter.requires_grad)
                gradients = torch.autograd.grad(value.sum(), parameters, retain_graph=True,
                                                allow_unused=True) if parameters else ()
                norm = sum(float(gradient.detach().double().square().sum())
                           for gradient in gradients if gradient is not None) ** .5
                self.counts['equality_trials_with_verb_shift_gradient'] += norm > 0.
            return value, counts

        def observed(model, meaning, **kwargs):
            row = kwargs.get('row', 0)
            store = model.symbolSpace.ltm_store
            start = len(store)
            value = run(model, meaning, **kwargs)
            self.counts['episodes'] += bool(open_slots(meaning))
            for record in value.records:
                if record.kind == 'thought':
                    self.operations[record.operation] += 1
            committed = [store.row(index) for index in range(start, len(store))
                         if store.KINDS[int(store.record_kind[index])] == 'inference']
            self.results[row] = (value, committed)
            return value

        def credited(model, value, *, costs, source):
            parameters = tuple(parameter for parameter in
                model._selected_thought_chooser(None).parameters() if parameter.requires_grad)
            gradients = (torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                         if value is not None and value.requires_grad else ())
            gradient_norm = sum(float(gradient.detach().double().square().sum())
                                for gradient in gradients if gradient is not None) ** .5
            item = dict(source=source, costs=list(map(float, costs)),
                        graph=value is not None and value.requires_grad,
                        surrogate=None if value is None else float(value.detach()),
                        raw_chooser_gradient_l2=gradient_norm)
            self.gradient_l2_sum += gradient_norm
            self.gradient_l2_max = max(self.gradient_l2_max, gradient_norm)
            self.credits[source] += 1
            self.credits['nonzero_graph'] += int(item['graph'] and item['surrogate'] != 0.)
            self.credits['ties'] += costs[0] == costs[1]
            self.credits['nonzero_chooser_gradients'] += gradient_norm > 0.
            if len(self.credit_sample) < 100:
                self.credit_sample.append(dict(self.context, **item))
            return register(model, value, costs=costs, source=source)

        self.stack.enter_context(patch.object(BasicModel, 'run_selected_thought', observed))
        self.stack.enter_context(patch.object(ThoughtCredit, 'register', credited))
        self.stack.enter_context(patch.object(EqualityLearning, 'cost', equality))
        return self

    def after_batch(self, model, split, source_rows, _result):
        from BindingAnswers import matches
        from ThoughtReferences import bindings, open_slots
        data = model.inputSpace.data
        fields = model._sentence_fields.get(0, ())
        store = model.symbolSpace.ltm_store
        for row, source in enumerate(source_rows):
            address = data.source_addresses[split][source]
            doc = data.math_chain_documents[split][address['document']]
            field = fields[row] if row < len(fields) else None
            if field is not None and field.row_id not in (-1, 0):
                index = store.index_of_row(field.row_id)
                if index is not None:
                    self.source_texts[store.occurrence_of(index)] = doc.sentences[address['sentence']]
            if address['sentence'] != doc.question:
                continue
            event = self.results.get(row)
            thought, inferences = event if event is not None else (None, [])
            meaning = thought.meaning if thought is not None else None if field is None else field.meaning
            committed_binding = inferences[-1] if inferences else None
            if thought is None and field is not None and field.row_id not in (-1, 0):
                index = store.index_of_row(field.row_id)
                if index is not None:
                    committed_binding = store.row(index)
            rows = []
            for item in inferences:
                value = item['meaning']
                metadata = bindings(value)
                witnesses = metadata.get('_thought_witnesses', ())
                rows.append(dict(occurrence=item['occurrence'],
                    refs=value.role_refs, open=open_slots(value),
                    operation=metadata.get('_producing_operation'),
                    witnesses=witnesses,
                    known_witnesses=bool(witnesses) and all(ref in store._index_occurrences for ref in witnesses),
                    witness_texts=[self.source_texts.get(ref) for ref in witnesses],
                    bound_words=[word for word in self.words if matches(model, value, word)]))
            exact_steps = len(rows) == len(doc.steps)
            previous = None
            for item, (left, one, right) in zip(rows, doc.steps):
                # A copied answer, a not-image or a right-length trace alone
                # is insufficient. Require the successor's bound identity,
                # the fact that licenses it, and dependency on the prior step.
                fact = f'{left} plus {one} is {right}.'
                licensed = any(text is not None and text.casefold() == fact
                               for text in item['witness_texts'])
                dependency = (previous in item['witnesses'] if previous is not None else
                    any(text == f'x is {left}.' for text in item['witness_texts']))
                exact_steps &= item['bound_words'] == [right] and item['known_witnesses'] and licensed and dependency
                previous = item['occurrence']
            correct = (committed_binding is not None and
                       matches(model, committed_binding['meaning'], doc.answer))
            item = dict(self.context, split=split, document=doc.key, pair=doc.pair,
                question_source=source, episode=thought is not None,
                work=0 if thought is None else thought.work.spent,
                answer=doc.answer, bound_correct=correct,
                committed_binding=None if committed_binding is None else committed_binding['occurrence'],
                open=None if meaning is None else open_slots(meaning),
                expected_chain_length=len(doc.steps), inference_count=len(rows),
                chain_correct=thought is not None and exact_steps and correct,
                inferences=rows,
                operations=[] if thought is None else [record.operation for record in thought.records if record.kind == 'thought'])
            self.questions.append(encode(item))
            self.question_log.write(json.dumps(encode(item)) + '\n')
            self.question_log.flush()
        self.counts['sentences'] += len(source_rows)
        self.results.clear()

    @property
    def words(self):
        from math_chain_corpus import NUMBER_WORDS
        return NUMBER_WORDS

    def report(self):
        return dict(counts=dict(self.counts), operations=dict(self.operations),
                    credit=dict(self.credits), credit_sample=self.credit_sample,
                    raw_chooser_gradient_l2_sum=self.gradient_l2_sum,
                    raw_chooser_gradient_l2_max=self.gradient_l2_max)

    def __exit__(self, *args):
        self.stack.close()
        self.question_log.close()
        (self.folder / 'observer.json').write_text(json.dumps(self.report(), indent=2) + '\n')
