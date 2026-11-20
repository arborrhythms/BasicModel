"""Teacher-owned primitive and name lessons for native conceptual output.

Only primitive memberships and single-occurrence names are taught here.
Located conjunctions come from the ordinary, target-free field observer;
output definitions begin with zero weights over every witnessed case.
"""
import json
from pathlib import Path

import torch

from util import TheXMLConfig, ProjectPaths


def read_native_forms(model, texts, encode):
    """Read the lesson's texts as sentences, by the path a reading takes,
    until MTM nominates each lexical form in them: one count under the form,
    a reading being one sentence of one stream. Returns the forms' rows."""
    cs, ws = model._concept_owner(), model.wholeSpace
    native = model.perceptualSpace.percept_store
    rows = {}
    for _ in range(cs.concept_mint_count):
        with torch.no_grad():
            model.forward(encode(texts))
        for b, text in enumerate(texts):
            raw = text.encode('utf8') if isinstance(text, str) else bytes(text)
            parts = tuple(native.spell_out(raw))
            word = cs.interpret.lookup_word(parts, ws.property_rows_for_bytes(raw), form=raw,
                                            word_reading=True, reading=cs.reading(b))
            if word is not None and word in cs.basic_percepts.words:
                rows[raw] = cs.basic_percepts.words[word]
        model.End()
    return tuple(rows.get(t.encode('utf8') if isinstance(t, str) else bytes(t), -1) for t in texts)


def teach_concept_lessons(model):
    filename = TheXMLConfig.get('architecture.data.conceptLessons', default='')
    if not filename or getattr(model, '_concept_lessons_taught', False):
        return
    path = Path(filename)
    if not path.is_absolute():
        path = Path(ProjectPaths.DATA_DIR) / path
    lesson = json.loads(path.read_text())
    cs, ws = model.conceptualSpaces[0], model.wholeSpace
    if not (cs._sparse_active() and cs._promotion_enabled):
        raise ValueError('concept lessons require symbolic order and promotion')
    primitive = ws.subspace.what.primitive_properties
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(cs)
    for item in lesson['properties']:
        primitive.teach(item['row'], item['bytes'], item['memberships'])
    # D2 names collect the basic percepts of the supplied training forms.
    # Admit those forms before constructing their sparse feature optimizer.
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

    forms = list(dict.fromkeys(example['text'] for item in lesson['names'] for example in item['examples']))
    was_training = model.training
    model.eval()
    read_native_forms(model, forms, encode)
    for item in lesson['names']:
        concept = int(item['concept'])
        if not item['features']:
            raise ValueError('a taught primitive name requires a percept definition')
        # This lesson supplies a definition over native percept memberships.
        # That testimony witnesses zero before the storage row is requested;
        # an arbitrary row request itself still cannot assign an order.
        cs._declare_concept_row(0, concept)
        alloc.settle(concept)
        row = cs._csw_concept_row(alloc.order_of(concept), concept)
        for feature in item['features']:
            cs.add_concept_feature(row, feature['tower'], feature['row'],
                                   feature['weight'], negated=feature.get('negative', False))
    def read_name(concept):
        carrier = model._combine_last_cs_sub
        ids, evidence = carrier._concept_ids, carrier._concept_activations
        if ids.ndim == 1:
            ids = ids[:, None].expand(-1, evidence.shape[1])
        return (evidence.amax(2) * (ids == concept)[..., None]).amax(0)

    model.set_sigma(0.)
    # Name targets never enter forward; optimize the names' entries of the
    # order-zero Sigma.
    optimizer = torch.optim.Adam([cs.order_zero.value], lr=float(lesson['learning_rate']))
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
            model.End()
        history.append(float(loss.detach()))

    # The cases are unlabelled training inputs. Observe their located field,
    # then advance the same kept-close clock and recurrence mint rule.
    inputs = model.inputSpace.data.train_input
    with torch.no_grad():
        for _ in range(int(lesson['witness_updates'])):
            for start in range(0, len(inputs), 64):
                texts = [x if isinstance(x, str) else
                         bytes(x.reshape(-1).tolist()).rstrip(b'\0').decode('utf8')
                         for x in inputs[start:start + 64]]
                model.forward(encode(texts))
                cs.observe_open_field()
                for stream in range(len(texts)):
                    cs.mtm.close(None, stream=stream, references=
                        getattr(cs, '_field_activated_references', {}).get(stream, ()))
                cs._field_activated_references = {}
                cs.mtm.promote_candidates(cs)
                model.End()
    cases = sorted({r for r in alloc.order_zero_members.rows()
                    if cs._order0_inventory_row(r) and cs.concept_id_at_row(r) in alloc.raised})
    if not cases:
        raise ValueError('concept lessons did not witness any located field conjunctions')
    orders = [alloc.order_of(cs.concept_id_at_row(row)) for row in cases]
    if min(orders) < 0:
        raise ValueError('an output definition requires witnessed source orders')
    model._concept_lesson_receipt = dict(name_losses=history, cases=cases)
    model._concept_lessons_taught = True
    model.zero_grad(set_to_none=True)
    model.train(was_training)
