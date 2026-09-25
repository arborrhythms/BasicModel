"""Teacher-owned primitive and name lessons for native conceptual output.

Only primitive memberships and single-occurrence names are taught here.
Located conjunctions come from the ordinary, target-free pool observer;
output definitions begin with zero weights over every witnessed case.
"""
import json
from pathlib import Path

import torch

from util import TheXMLConfig, ProjectPaths


def teach_concept_lessons(model):
    filename = TheXMLConfig.get('architecture.data.conceptLessons', default='')
    if not filename or getattr(model, '_concept_lessons_taught', False):
        return
    path = Path(filename)
    if not path.is_absolute():
        path = Path(ProjectPaths.DATA_DIR) / path
    lesson = json.loads(path.read_text())
    cs, ws = model.conceptualSpaces[0], model.wholeSpaces[0]
    if not (cs._sparse_active() and cs.conceptual_pi and cs._promotion_enabled):
        raise ValueError('concept lessons require symbolic order, conceptual pi and promotion')
    primitive = ws.subspace.what.primitive_properties
    for item in lesson['properties']:
        primitive.teach(item['row'], item['bytes'], item['memberships'])
    for item in lesson['names']:
        row = cs._csw_concept_row(0, int(item['concept']))
        for feature in item['features']:
            cs.add_concept_feature(row, feature['tower'], feature['row'],
                                   feature['weight'], negated=feature.get('negative', False))
    for concept in model.outputSpace.concept_ids:
        cs._csw_concept_row(1, concept)
    store = cs._concept_allocator.layer(0)
    device = primitive.members.device
    width = int(model.inputSpace.outputShape[0])

    def encode(texts):
        data = torch.zeros(len(texts), 1, width, device=device, dtype=torch.long)
        for b, text in enumerate(texts):
            raw = text.encode('utf8')
            if len(raw) > width:
                raise ValueError('concept lesson exceeds input width')
            data[b, 0, :len(raw)] = torch.tensor(list(raw), device=device)
        return data

    def read_name(concept):
        carrier = model._combine_last_cs_sub
        ids, evidence = carrier._concept_ids, carrier._concept_activations
        if ids.ndim == 1:
            ids = ids[:, None].expand(-1, evidence.shape[1])
        return (evidence.amax(2) * (ids == concept)[..., None]).amax(0)

    was_training = model.training
    model.eval()
    model.set_sigma(0.)
    # Name targets never enter forward; optimize the existing feature weights.
    optimizer = torch.optim.Adam([store.features.values], lr=float(lesson['learning_rate']))
    history = []
    for item in lesson['names']:
        examples = item['examples']
        native = encode([e['text'] for e in examples])
        target = torch.tensor([e['evidence'] for e in examples], device=device)
        for _ in range(int(lesson['updates'])):
            optimizer.zero_grad()
            model.forward(native)
            loss = (read_name(item['concept']) - target).square().mean()
            loss.backward()
            optimizer.step()
            store.project_parts()
            model.End()
        history.append(float(loss.detach()))

    # The cases are unlabelled training inputs. Observe their located field,
    # then let raw use (never the optimizer) earn pool participation.
    inputs = model.inputSpace.data.train_input
    with torch.no_grad():
        for _ in range(int(lesson['witness_updates'])):
            for start in range(0, len(inputs), 64):
                texts = [x if isinstance(x, str) else
                         bytes(x.reshape(-1).tolist()).rstrip(b'\0').decode('utf8')
                         for x in inputs[start:start + 64]]
                model.forward(encode(texts))
                cs.promotion_observe()
                cs.promotion_pass()
                model.End()
    cases = sorted({r for r, _ in store.conjunctive._index
                    if cs._order0_inventory_row(r) and not bool(store.provisional[r])})
    if not cases:
        raise ValueError('concept lessons did not witness any located field conjunctions')
    for concept in model.outputSpace.concept_ids:
        output = cs._csw_row_of(concept)
        for row in cases:
            cs.add_concept_edge(output, row)  # unwritten; labels train these later
    model._concept_lesson_receipt = dict(name_losses=history, cases=cases,
                                        output_concepts=list(model.outputSpace.concept_ids))
    model._concept_lessons_taught = True
    model.zero_grad(set_to_none=True)
    model.train(was_training)
