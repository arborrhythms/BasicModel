"""Matched CE on actual student states from the pilot's first paired batch."""
import torch

from GrammarLessons import (constituent_examples, score_student_state,
                           student_target_actions, student_target_probability)
from identity_experiment import save


class PilotProbe:
    def __init__(self, model, output):
        self.model, self.output = model, output
        self.states = []
        self.before_values = {'wait': [], 'fold': []}
        self.initial_pressure = float(model.languageSpace._tree_layer(2).reduce_pressure.detach())
        self.collecting = True

    def capture(self, language, programs, lessons):
        if not self.collecting:
            return
        for program, lesson in zip(programs, lessons):
            for state in constituent_examples(language, program, lesson['tree']):
                self.states.append(dict(state, text=lesson['text']))
                target = state['target']
                self.before_values['wait' if target == -1 else 'fold'].append(
                    float(-student_target_probability(language._tree_layer(2), state,
                        state['observed_probabilities']).clamp_min(1e-30).log()))

    @staticmethod
    def summary(values, pressure):
        return dict(pressure=pressure, **{name: dict(count=len(rows),
            ce=sum(rows)/len(rows) if rows else None) for name, rows in values.items()})

    def finish(self):
        if not self.states:
            raise RuntimeError('pilot received no student states')
        layer = self.model.languageSpace._tree_layer(2)
        values = {'wait': [], 'fold': []}
        diagnostics = []
        with torch.no_grad(), torch.random.fork_rng():
            for state in self.states:
                probabilities = score_student_state(layer, state)
                target = state['target']
                probability = student_target_probability(layer, state, probabilities)
                value = float(-probability.clamp_min(1e-30).log())
                values['wait' if target == -1 else 'fold'].append(value)
                diagnostics.append(dict(text=state['text'], slot=state['slot'], spans=state['spans'],
                    target=target, target_actions=student_target_actions(layer, state), selected_kind=state['kind'],
                    before=float(-student_target_probability(layer, state,
                        state['observed_probabilities']).clamp_min(1e-30).log()),
                    after=value, unreachable=not bool(probability > 0)))
        before = self.summary(self.before_values, self.initial_pressure)
        after = self.summary(values, float(layer.reduce_pressure.detach()))
        passed = all(before[name]['count'] and before[name]['ce']-after[name]['ce'] > 1e-5
                     for name in ('wait', 'fold'))
        result = dict(before=before, after=after, passed=bool(passed),
            gate='both raw mean CEs on first-batch student states decrease by more than 1e-5 nats',
            texts=list(dict.fromkeys(state['text'] for state in self.states)))
        save(self.output/'student-states.json', diagnostics)
        save(self.output/'pilot.json', result)
        return result
