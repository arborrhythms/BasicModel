"""Fixed teacher-state CE probe inside the first native training presentation."""
import torch

from GrammarLessons import constituent_examples
from identity_experiment import save


class PilotProbe:
    def __init__(self, model, output):
        self.model, self.output = model, output
        self.states = []
        self.before = None

    def capture(self, language, programs, lessons):
        if self.before is not None:
            return
        with torch.no_grad(), torch.random.fork_rng():
            for program, lesson in zip(programs, lessons):
                for state in constituent_examples(language, program, lesson['tree']):
                    if state.get('unavailable'):
                        raise RuntimeError('pilot teacher target unavailable')
                    self.states.append(dict(state, capacity=program.compose_capacity, text=lesson['text']))
            self.before = self.measure()
            torch.save(self.states, self.output/'teacher-states.pt')
            save(self.output/'ce-before.json', self.before)

    def measure(self):
        layer = self.model.languageSpace._tree_layer(2)
        width = int(getattr(layer.chooser, 'WHAT_CONTEXT_DIM', 0))
        values = {'wait': [], 'fold': []}
        with torch.no_grad(), torch.random.fork_rng():
            for state in self.states:
                window, depth, closing = state['window'], state['depth'], state['closing']
                _, _, route = layer(window[None],
                    depth=torch.tensor([min(2, depth)], device=window.device),
                    slots=2 if not closing or depth == 1 else 0,
                    allowance=1 if closing else state['capacity']-1,
                    rounds_left=state['remaining'],
                    load_depth=torch.tensor([depth], device=window.device),
                    reference_data=state['references'], grammar_context=state['context'],
                    what_ctx=window.new_zeros(1, width))
                target = state['target']
                values['wait' if target < 0 else 'fold'].append(
                    float(-route['probabilities'][0, target].clamp_min(1e-30).log()))
        return dict(pressure=float(layer.reduce_pressure.detach()),
            **{name: dict(count=len(rows), ce=sum(rows)/len(rows)) for name, rows in values.items()})

    def finish(self):
        if self.before is None:
            raise RuntimeError('pilot received no teacher states')
        after = self.measure()
        # The declared pilot gate requires both means to fall, beyond float noise.
        passed = all(self.before[name]['ce']-after[name]['ce'] > 1e-5 for name in ('wait', 'fold'))
        result = dict(before=self.before, after=after, passed=passed,
            gate='both fixed-state raw mean CEs decrease by more than 1e-5 nats',
            texts=list(dict.fromkeys(state['text'] for state in self.states)))
        save(self.output/'pilot.json', result)
        return result
