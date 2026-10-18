"""The existing trajectory plus the unchanged reference menu's first-pair scores."""
import torch

from constituent_trajectory import TrainingTrajectory
from GrammarLessons import ComposeLessonStates, constituent_examples


class NPTrajectory(TrainingTrajectory):
    def __init__(self, patch):
        super().__init__(patch)
        original = ComposeLessonStates.record
        def record(capture, slot, window, kwargs, route):
            original(capture,slot,window,kwargs,route)
            for row in route['valid'].nonzero().flatten().tolist():
                key=int(slot[row] if torch.is_tensor(slot) and slot.ndim else slot)
                capture.rounds[row,key]['observed_logits']=route['logits'][row].detach().clone()
        patch.setattr(ComposeLessonStates,'record',record)

    def capture(self, language, programs, lessons):
        super().capture(language,programs,lessons)
        layer=language._tree_layer(2)
        for row,program,lesson in zip(self.calls[-1],programs,lessons,strict=True):
            for state in constituent_examples(language,program,lesson['tree']):
                if state['spans'] != ((0,),(1,)):
                    continue
                references=state['kwargs'].get('reference_data')
                menu=tuple(range(layer.r_reduce)) if references is None else references['binary_ops']
                logits,probabilities=state['observed_logits'],state['observed_probabilities']
                occupation=state['kwargs']['occupation'][state['row']]
                scores=[]
                for op in sorted(set(menu)):
                    indices=[i for i,value in enumerate(menu) if value==op]
                    values=logits[indices]
                    finite=values[torch.isfinite(values)]
                    scores.append(dict(operation=layer.argument_operation_names[op],
                        variants=len(indices),legal=int(finite.numel()),
                        maximum_logit=float(finite.max()) if finite.numel() else None,
                        occupation_term=float(occupation[0,layer.argument_inputs[op,0]]+
                                              occupation[1,layer.argument_inputs[op,1]]),
                        probability=float(probabilities[indices].sum())))
                scores.append(dict(operation='STOP',variants=1,legal=int(torch.isfinite(logits[-1])),
                    maximum_logit=float(logits[-1]) if torch.isfinite(logits[-1]) else None,
                    occupation_term=0.,
                    probability=float(probabilities[-1])))
                row['first_pair']['menu']=scores
                break
