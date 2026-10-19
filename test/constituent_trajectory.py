"""Read-only trajectory on training trials and their selected kept readings."""
from collections import Counter
import math
from GrammarLessons import target_operations

from constituent_corpus import FAMILY_ORDER


def summarize_batches(batches):
    calls = sum(row['lesson_calls'] for row in batches)
    folds = sum(row['fold_targets'] for row in batches)
    ce = {}
    for target in ('wait', 'fold'):
        count = sum(row['ce'][target]['reachable'] for row in batches)
        total = sum(row['ce'][target]['sum'] for row in batches)
        ce[target] = dict(reachable=count, mean=total/count if count else None,
            unreachable=sum(row['ce'][target]['unreachable'] for row in batches))
    kept = [row for batch in batches for row in batch['kept']]
    families = {}
    for family in FAMILY_ORDER:
        rows = [row['first_pair'] for row in kept if row['family'] == family]
        correct = sum(row is not None and row['correct'] for row in rows)
        families[family] = dict(total=len(rows), correct=correct,
            unreached=sum(row is None for row in rows), accuracy=correct/len(rows) if rows else None)
    steps = {name: sum(row['steps'][name] for row in kept)
             for name in ('leaf', 'binary', 'unary', 'transpose', 'compose', 'total')}
    return dict(first_batch=batches[0]['batch'], last_batch=batches[-1]['batch'],
        lesson_calls=calls, fold_targets=folds, fold_targets_per_lesson_call=folds/calls,
        reachable_state_ce=ce, first_pair_by_family=families,
        kept_program_steps=dict(sentences=len(kept), totals=steps,
            means={name: value/len(kept) for name,value in steps.items()}))


class TrainingTrajectory:
    """Capture loss inputs without rescoring, changing RNG, or teaching a parse."""
    def __init__(self, patch):
        import GrammarLessons
        record = GrammarLessons.ComposeLessonStates.record
        compose = GrammarLessons.compose_loss
        self.calls, self.batches = [], []

        def record_operation(capture, slot, window, kwargs, route):
            record(capture, slot, window, kwargs, route)
            import torch
            for row in route['valid'].nonzero().flatten().tolist():
                key = int(slot[row] if torch.is_tensor(slot) and slot.ndim else slot)
                capture.rounds[row, key]['operation'] = int(route['op'][row])

        def observe(language, programs, lessons, **kwargs):
            self.capture(language, programs, lessons)
            return compose(language, programs, lessons, **kwargs)

        patch.setattr(GrammarLessons.ComposeLessonStates, 'record', record_operation)
        patch.setattr(GrammarLessons, 'compose_loss', observe)

    def capture(self, language, programs, lessons):
        from GrammarLessons import constituent_examples, student_target_probability
        layer = language._tree_layer(2)
        call = []
        for program, lesson in zip(programs, lessons, strict=True):
            states = constituent_examples(language, program, lesson['tree'])
            ce = {name: dict(reachable=0, sum=0., unreachable=0) for name in ('wait', 'fold')}
            first = None
            for state in states:
                target = state['target']
                group = ce['wait' if target == -1 else 'fold']
                probability = float(student_target_probability(layer, state, state['observed_probabilities']))
                if probability > 0:
                    group['reachable'] += 1
                    group['sum'] -= math.log(probability)
                else:
                    group['unreachable'] += 1
                if first is None and state['spans'] == ((0,), (1,)):
                    first = dict(target=target, kind=state['kind'], operation=state['operation'],
                        correct=(state['kind'] == 0 if target == -1 else
                                 state['kind'] == 1 and state['operation'] in target_operations(target)))
            call.append(dict(text=lesson['text'], fold_targets=sum(state['target'] != -1 for state in states), ce=ce,
                             first_pair=first))
        self.calls.append(call)

    def finish_batch(self, number, epoch, rows, observations, wins):
        calls, self.calls = self.calls, []
        if len(calls) != 2*len(rows) or any(len(call) != 1 for call in calls):
            raise ValueError('trajectory requires one lesson call per row in each native trial')
        if any(call[0]['text'] != rows[i % len(rows)]['text'] for i,call in enumerate(calls)):
            raise ValueError('trajectory lesson calls differ from native trial/row order')
        ce = {name: {key: sum(row['ce'][name][key] for call in calls for row in call)
                     for key in ('reachable', 'sum', 'unreachable')} for name in ('wait', 'fold')}
        kept = []
        for b, row in enumerate(rows):
            observed = observations.get((0, b))
            if observed is None:
                raise ValueError('training trajectory missed a kept reading')
            counts = Counter(step['kind'] for step in observed['steps'])
            counts = {name: counts[name] for name in ('leaf', 'binary', 'unary', 'transpose')}
            counts['compose'] = sum(counts[name] for name in ('binary', 'unary', 'transpose'))
            counts['total'] = counts['leaf'] + counts['compose']
            kept.append(dict(id=row['id'], family=row['id'].split(':')[1],
                trial=int(bool(wins[b])), first_pair=calls[int(bool(wins[b]))*len(rows)+b][0]['first_pair'], steps=counts))
        self.batches.append(dict(batch=number, epoch=epoch, lesson_calls=len(calls), ce=ce,
            fold_targets=sum(row['fold_targets'] for call in calls for row in call), kept=kept))

    def report(self):
        return dict(blocks=[summarize_batches(self.batches[start:start+8])
                           for start in range(0, len(self.batches), 8)],
            definitions=dict(ce='observed pre-update probabilities on reachable student trial-states; reused prefixes count in each loss call',
                first_pair='first visited ((0,), (1,)) source pair in the kept trial; missing pairs count as incorrect',
                steps='chronological kept leaf, binary, unary and transpose actions; compose excludes leaves'),
            overall=summarize_batches(self.batches))
