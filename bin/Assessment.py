"""Read native sentence trees without retaining any effect of the probe."""

from ReadingIndex import steps as recorded_steps
from contextlib import contextmanager
from pathlib import Path
import json

import torch


class _State:
    """An in-process evaluation transaction, including non-checkpoint scratch.

    Checkpoint loading cannot restore live graph identities or an open MTM
    frame. Retain the original objects and containers and restore their data
    instead. Only objects reachable from the model are visited; modules,
    functions and classes are not traversed into process globals.
    """
    def __init__(self, model):
        import random
        import types
        import weakref
        from collections import deque
        from Layers import _RadixNode
        import numpy as np
        self.random = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        from Models import _mps_rng_state_or_none
        self.mps = _mps_rng_state_or_none()
        from MemoryActivity import snapshot
        self.activity = snapshot()
        self.saved, seen = [], set()

        def visit(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if torch.is_tensor(value):
                if value._base is not None:
                    visit(value._base)
                self.saved.append(('tensor', value, value.detach().clone()))
            elif isinstance(value, _RadixNode):
                # Radix nodes are slotted. Restoring just the trie's count
                # leaves the previous presentation's terminals in its tree.
                self.saved.append(('radix_node', value, (value.prefix, value.percept_id, value.children)))
                visit(value.children)
            elif isinstance(value, np.ndarray):
                self.saved.append(('array', value, value.copy()))
            elif isinstance(value, torch.Generator):
                self.saved.append(('generator', value, value.get_state()))
            elif isinstance(value, weakref.ReferenceType):
                visit(value())
            elif isinstance(value, dict):
                self.saved.append(('dict', value, dict(value)))
                for item in tuple(value.values()):
                    visit(item)
            elif isinstance(value, (list, deque, set)):
                self.saved.append(('sequence', value, tuple(value)))
                for item in value:
                    visit(item)
            elif isinstance(value, (tuple, frozenset, types.MappingProxyType)):
                for item in value.values() if isinstance(value, types.MappingProxyType) else value:
                    visit(item)
            elif (hasattr(value, '__dict__') and not isinstance(value,
                    (type, types.ModuleType, types.FunctionType, types.MethodType))):
                # Library modules are owned through their Module dictionaries;
                # Child optimizers own the moments that move when STM/MTM
                # reallocates a parameter during a probe.
                module = type(value).__module__
                if isinstance(value, (torch.nn.Module, torch.optim.Optimizer)) or module.split('.')[0] in (
                        'types', 'collections') or (Path(__file__).parent / (module.split('.')[-1]+'.py')).is_file():
                    self.saved.append(('object', value, dict(vars(value))))
                    for item in tuple(vars(value).values()):
                        visit(item)
        visit(model)

    @torch.no_grad()
    def restore(self):
        import random
        import numpy as np
        for kind, target, value in self.saved:
            if kind == 'tensor':
                same = target.shape == value.shape and target.layout == value.layout
                if same and target.layout == torch.strided:
                    same = torch.equal(target, value)
                elif same and target.layout == torch.sparse_coo:
                    a, b = target.coalesce(), value.coalesce()
                    same = torch.equal(a.indices(), b.indices()) and torch.equal(a.values(), b.values())
                else:
                    same = False
                if not same:
                    if target.shape == value.shape and target.layout == value.layout:
                        target.copy_(value)
                    else:
                        target.data = value.clone()
            elif kind == 'radix_node':
                target.prefix, target.percept_id, target.children = value
            elif kind == 'array':
                target[...] = value
            elif kind == 'generator':
                target.set_state(value)
            elif kind == 'object':
                vars(target).clear()
                vars(target).update(value)
            elif kind == 'dict':
                target.clear()
                target.update(value)
            elif kind == 'sequence':
                target.clear()
                (target.update if isinstance(target, set) else target.extend)(value)
        random.setstate(self.random[0])
        np.random.set_state(self.random[1])
        torch.random.set_rng_state(self.random[2])
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)
        if self.mps is not None:
            torch.mps.set_rng_state(self.mps)
        from MemoryActivity import restore
        restore(self.activity)


@contextmanager
def preserved(model):
    """Ordinary reads, with all their model-owned effects rolled back."""
    state = _State(model)
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        state.restore()


def reading_forest(language, program, positions):
    """Read actual journal structure, preserving source places and bind choices."""
    if program is None:
        return []
    stack = []
    for action, (kind, local, word) in enumerate(recorded_steps(program)):
        if kind < 0:
            break
        if kind == 0:
            stack.append(int(positions[word]))
        elif kind == 3:
            stack[-2:] = reversed(stack[-2:])
        elif kind in (1, 2):
            rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
            rule = rules[local]
            name = getattr(rule, 'surface_name', None) or rule.method_name
            if kind == 1 and rule.method_name == 'bind':
                opened = program.operation_refs is not None and int(program.operation_refs[action, 1]) == 0
                stack[-1] = ['bind.open' if opened else 'bind', stack[-1]]
            elif kind == 1:
                right, left = stack.pop(), stack.pop()
                stack.append([name, left, right])
            else:
                position = len(stack)-1-max(0, word)
                stack[position] = [name, stack[position]]
        else:
            raise ValueError('unknown native reading action')
    return stack


def named_tree(tree, words):
    """Compact successive unary uses without losing their count or opening."""
    if isinstance(tree, int):
        return dict(position=tree, word=next(word['form'] for word in words if word['position'] == tree))
    name, *children = tree
    count = 1
    while len(children) == 1 and isinstance(children[0], list) and children[0][0] == name:
        count += 1
        children = children[0][1:]
    return dict(operator=name, count=count, open=name == 'bind.open',
                children=[named_tree(child, words) for child in children])


def tree_text(tree):
    if 'position' in tree:
        return tree['word']
    name = tree['operator'] + (f" x{tree['count']}" if tree['count'] > 1 else '')
    return name+'('+', '.join(map(tree_text, tree['children']))+')'


@contextmanager
def _native_word_fields(model):
    """Attribute the field attention actually reads, without reading it again.

    These are the native input lanes at each word's extent, before any walk
    reduction. Retain all positive maximum contributors, including ties;
    the complete per-concept pairs distinguish shared from separate sources.
    """
    import ModelAttention
    observations = []
    original = ModelAttention.native_word_poles

    def observed(current, spans, forms, known):
        if current is not model:
            return original(current, spans, forms, known)
        owner = model._concept_owner()
        read = owner.cs_read_memberships
        had_override = 'cs_read_memberships' in vars(owner)
        override = vars(owner).get('cs_read_memberships')
        captured = []

        def field_read(*args, **kwargs):
            field = read(*args, **kwargs)
            captured.append((field.detach().cpu().clone(),
                owner._cs_field_concept_ids[:field.shape[0]].detach().cpu().clone()))
            return field

        owner.cs_read_memberships = field_read
        try:
            poles = original(current, spans, forms, known)
        finally:
            if had_override:
                owner.cs_read_memberships = override
            else:
                del owner.cs_read_memberships
        readings = [[None for _ in row] for row in forms]
        if captured:
            field, ids = captured[-1]
            for batch, row in enumerate(forms):
                native = ids if ids.ndim == 1 else ids[:, batch]
                for position, form in enumerate(row[:field.shape[2]]):
                    if not bool(known[batch, position]):
                        continue
                    word = owner.interpret.word_for_form(form)
                    concepts = []
                    for concept in dict.fromkeys((word, *owner.definitions.objects(word))):
                        selected = native == concept
                        pair = (field[selected, batch, position].amax(0) if bool(selected.any())
                                else field.new_zeros(2))
                        concepts.append(dict(concept=concept, row=owner._concept_row_of(concept),
                            role='word' if concept == word else 'object', evidence=pair.tolist()))
                    lanes = {}
                    for lane, name in enumerate(('for', 'against')):
                        value = float(poles[batch, position, lane])
                        lanes[name] = dict(value=value, sources=[
                            {key: entry[key] for key in ('concept', 'row', 'role')}
                            for entry in concepts if value > 0 and entry['evidence'][lane] == value])
                    readings[batch][position] = dict(span=spans[batch, position].tolist(),
                        concepts=concepts, lanes=lanes)
        observations.append(readings)
        return poles

    ModelAttention.native_word_poles = observed
    try:
        yield observations
    finally:
        ModelAttention.native_word_poles = original


@contextmanager
def _attention_trace(model):
    """Observe each look's request and what it returned, without choosing
    or replaying an act."""
    import SentenceLooks
    from SentenceLooks import Looks
    walks, current = [], []
    run, requested = Looks.run, SentenceLooks.attend

    def observed_request(model_, units):
        result = requested(model_, units)
        if current and model_ is model:
            asked = (result[..., 0] > 0).nonzero().tolist()
            current[-1]['requests'].append([[row, stream] for row, stream in asked])
        return result

    def observed_run(path, arrive, **kwargs):
        if path.model is not model:
            return run(path, arrive, **kwargs)
        state = path.walk
        item = dict(requests=[], limit=state.limit.tolist(), spans=state.spans.tolist())
        current.append(item)
        try:
            result = run(path, arrive, **kwargs)
            item.update(actions=result.actions.tolist(), accepted=result.accepted.tolist(),
                        descended=result.descended.tolist(), looks=result.looks.tolist(), turns=[
                            dict(extent=field['extent'].tolist(), active=field['active'].tolist(),
                                 surprise=None if field['surprise'] is None else field['surprise'].tolist(),
                                 word=decision['index'].tolist(), offered=decision['logits'][:,1].isfinite().tolist(),
                                 interpreted=decision['interpreted'].tolist())
                            for field,decision in zip(path.fields[1:],path.decisions)])
            walks.append(item)
            return result
        finally:
            current.pop()

    Looks.run, SentenceLooks.attend = observed_run, observed_request
    try:
        yield walks
    finally:
        Looks.run, SentenceLooks.attend = run, requested


def _references(meaning):
    if meaning is None:
        return None
    from ThoughtReferences import open_slots
    return dict(references=list(meaning.role_refs), open_references=list(open_slots(meaning)),
                children=[_references(child) for child in meaning.constituents])


def read_sentence(model, text, *, scalar=None, context=(), answer=False, trace=False):
    """Read one sentence, optionally in temporary discourse, and restore it all.

    The caller supplies text (and a felt value), never a slot, meaning, tree
    or answer reference. Expected probe trees are compared only after this
    function has returned.
    """
    from contextlib import nullcontext
    from util import parse
    from ConceptTrust import definition_threshold
    with preserved(model):
        context_meanings, context_subjects = [], []
        for prior in context:
            earlier = model.understand(model.inputSpace.prepInput([prior]))
            context_meanings.extend(_references(field.meaning) for field in earlier.sentence_states if field is not None)
            # The proposed context frame puts its name at source place zero.
            # Read that name's actual denotation, not the shared verb's ref.
            form = next((form.strip() for form, _ in parse(prior, lex='words') if form.strip()), None)
            interpret = model._concept_owner().interpret
            word = None if form is None else interpret.word_for_form(form)
            context_subjects.extend(('sym', obj) for obj in (() if word is None else interpret.denotations(word)))
            model.End()
        owner = model._concept_owner()
        # The same native lexical splitter supplies the pre-read lookup.
        before = {form.strip(): owner.interpret.word_for_form(form.strip())
                  for form, _ in parse(text, lex='words') if form.strip()}
        records = []
        original = model._sentence_observation
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get('admit'):
                record = model._trial_understanding(*args[:3])
                records[:] = [(reading_forest(model.languageSpace, program,
                    model._sentence_leaf_positions(record, row).tolist()),
                    _references(result['meanings'][row]))
                    for row, program in enumerate(result['entries'])]
            return result
        model._sentence_observation = observed
        with (_native_word_fields(model) as native_fields,
              _attention_trace(model) if trace else nullcontext([]) as walks):
            source = text if scalar is None else dict(text=text, scalar=scalar)
            understood = model.understand(model.inputSpace.prepInput([source]))
            forms = getattr(model, '_attention_forms', None)
            forms = (forms[0][0] if forms is not None else
                     (model.perceptualSpace._forward_input.get('word_texts') or [[]])[0])
            reading = getattr(model, '_attention_words', None)
            words = []
            store = owner._closed_clause_store()
            for position, raw in enumerate(forms):
                form = raw.strip()
                if not form:
                    continue
                word = owner.interpret.word_for_form(form)
                objects = () if word is None else owner.interpret.denotations(word)
                speech = model.languageSpace.parts_of_speech
                fits = () if word is None else speech.similarities(word).detach().tolist()
                arguments = sorted(zip(speech.arguments, fits), key=lambda item: (-item[1], item[0]))
                evidence = [store.parts.evidence(owner.definitions.row(word, obj)) for obj in objects]
                attention = (None if reading is None else 'interpreted' if bool(reading.accepted[0, position])
                             else 'entered' if bool(reading.descended[0, position]) else 'left')
                words.append(dict(position=position, form=form, known=before.get(form) is not None,
                    word=word, word_row=None if word is None else owner._concept_row_of(word),
                    objects=list(objects), definition_evidence=[[float(v) for v in pair] for pair in evidence],
                    trusted=bool(evidence) and all(float(pair[0]) >= definition_threshold() for pair in evidence),
                    attention=attention,
                    nearest_arguments=[dict(argument=name, fit=float(fit)) for name, fit in arguments[:3]],
                    native_field=native_fields[-1][0][position] if native_fields else None))
            forest, meaning = records[0] if records else ([], None)
            def leaves(node):
                return [node] if isinstance(node, int) else [i for child in node[1:] for i in leaves(child)]
            present = [i for tree in forest for i in leaves(tree)]
            trees = [named_tree(tree, words) for tree in forest]
            result = dict(text=text, words=words, forest=forest, trees=trees,
                missing=[word['position'] for word in words if word['position'] not in present],
                meaning=meaning, context_meanings=context_meanings, context_subjects=context_subjects,
                answer=None, evidence_kind=None)
            if reading is not None:
                taken = [a for a in reading.actions[0].tolist() if a >= 0 and bool(reading.accepted[0, a])]
                result['order'] = list(dict.fromkeys(taken))
                selected = [walk for walk in walks if walk['actions'] == reading.actions.tolist()]
                result['attention'] = dict(accepted=reading.accepted[0].tolist(),
                    descended=reading.descended[0].tolist(), actions=reading.actions[0].tolist(),
                    looks=int(reading.looks[0]),
                    trace=selected[-1] if selected else None)
            if answer:
                from What import What
                from ThoughtReferences import bindings
                derived = model.resolveAnswer(understood, What.inference(0, prompt=text))
                emitted = model.reverseOutput(understood, derived)
                result['answer'] = emitted.texts[0]
                result['answer_references'] = [_references(m) for m in derived.answer_meanings[0]]
                result['evidence_kind'] = [bindings(m).get('_evidence_kind') for m in derived.answer_meanings[0]]
            return result


def format_record(record):
    words = '  '.join(word['form']+'['+', '.join((
        'known' if word['known'] else 'unknown', word['attention'] or 'no walk'))+']' for word in record['words'])
    missing = ', '.join(word['form'] for word in record['words'] if word['position'] in record['missing']) or 'none'
    sources = []
    for word in record['words']:
        arguments = word.get('nearest_arguments', ())
        if arguments:
            sources.append('  arguments '+word['form']+': '+', '.join(
                item['argument']+' '+format(item['fit'], '.6g') for item in arguments))
        field = word.get('native_field')
        if (word['attention'] != 'left' or field is None
                or not all(lane['value'] > 0 for lane in field['lanes'].values())):
            continue
        lanes = [name+' '+str(lane['value'])+' from '+', '.join(
            source['role']+' '+str(source['concept'])+' (row '+str(source['row'])+')'
            for source in lane['sources']) for name, lane in field['lanes'].items()]
        sources.append('  native  '+word['form']+' at '+str(field['span'])+': '+'; '.join(lanes))
    return '\n'.join((json.dumps(record['text'], ensure_ascii=False), '  words   '+words,
        '  tree    '+(' ; '.join(map(tree_text, record['trees'])) or '(empty)'),
        '  missing '+missing, *sources, *(() if record['answer'] is None else
             ('  answer  '+record['answer']+' '+str(record['evidence_kind']),))))



def _answer(model, case, index, answers):
    """Present a supervised pair. The target belongs to Data.what alone."""
    from What import What
    with preserved(model):
        data = model.inputSpace.data
        for prior in case.context:
            model.understand(model.inputSpace.prepInput([prior]))
            model.End()
        data.has_supervised_outputs = True
        data.text_answers = {'train': answers}
        question = What.supervised(index, split='train', prompt=case.prompt)
        source = model.inputSpace.prepInput([case.text])
        batch, _ = model.runBatch(train=False, optimizer=None, batchSize=1, split='train',
            questions=(question,), batch_override=(source, torch.empty(1, 0)))
        returned = None if batch is None else batch.outputPred
        if returned is None or returned.is_floating_point():
            return None
        return bytes(returned.detach().reshape(-1).tolist()).rstrip(b'\0').decode('latin1')


def _xor_result(model):
    """Use the standing gate's four inputs and its MSE bar on these weights."""
    from AgeChart import xor_mse_bar
    with preserved(model):
        pairs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        source = model.inputSpace.prepInput(list(pairs))
        target = model.outputSpace.prepOutput([torch.tensor([v]) for v in (0., 1., 1., 0.)])
        batch, _ = model.runBatch(train=False, optimizer=None, batchSize=4, split='train',
                                  batch_override=(source, target))
        if batch is None or batch.outputPred is None:
            return dict(status='failed', returned=None)
        actual = batch.outputPred.detach()
        if actual.numel() != 4:
            return dict(status='failed', returned=actual.tolist())
        mse = float((actual.reshape(4) - target.reshape(4).to(actual)).square().mean())
        return dict(status='passed' if mse < xor_mse_bar() else 'failed', returned=actual.tolist(), mse=mse)


def _text_row(model, cases):
    from Spaces import SentenceTooLong
    records = []
    answers = [case.answer for case in cases]
    for index, case in enumerate(cases):
        record = dict(input=case.text, prompt=case.prompt, context=case.context, expected=case.answer)
        try:
            record['returned'] = _answer(model, case, index, answers)
        except SentenceTooLong as error:
            record['unavailable'] = str(error)
        records.append(record)
    unavailable = sum('unavailable' in case for case in records)
    passed = sum(case.get('returned') == case['expected'] for case in records)
    if passed + unavailable < len(records):
        status = 'failed'
    else:
        status = 'unavailable' if unavailable else 'passed'
    return dict(status=status, passed=passed, total=len(records), unavailable=unavailable, cases=records)


def units_across_modality(model):
    """The units across modality the model has learned, counted by order
    (``{order: count}``): the concepts that join concepts of two modalities,
    as a name joins its word and its object (``FamiliarTier.units``)."""
    from collections import Counter
    from FamiliarTier import units
    return dict(sorted(Counter(order for _, order, _ in units(model._concept_owner())).items()))


def _attained_age(rows):
    """Award complete, consecutive age groups in the measured chart."""
    if not rows:
        return 0.
    measured = {row['age'] for row in rows}
    first_unmet = min((row['age'] for row in rows if row['status'] != 'passed'), default=float('inf'))
    return max((years for years in measured if years < first_unmet), default=0.)


# The chart presents its own batch sizes on a model whose training bricks
# may already have a fixed compiled width.
@torch.compiler.set_stance('force_eager')
def assess_age(model, *, all_rows=True):
    """Assess this instance without training. Only this function awards an age.

    The full report measures higher abilities even after a lower row fails.
    With all_rows=False, stop after the first disqualifying age group. The
    age awarded is identical for the same observations of that prefix.
    """
    from itertools import groupby
    from AgeChart import chart
    text_model = model.inputSpace.model_type == 'embedding'
    xor_model = (not text_model and tuple(model.inputSpace.inputShape) == (2, 1)
                 and tuple(model.outputSpace.outputShape) == (1, 1))
    rows = []
    for years, group in groupby(chart(), key=lambda row: row[0]):
        first = len(rows)
        for _, name, cases in group:
            if name == 'exclusive or' and xor_model:
                result = _xor_result(model)
            elif cases and text_model:
                result = _text_row(model, cases)
            else:
                result = dict(status='unavailable', reason='requires ' + ('text input' if cases else 'the two-input XOR presentation'))
            rows.append(dict(age=years, measure=name, **result))
        if not all_rows and any(row['status'] != 'passed' for row in rows[first:]):
            break
    earned = _attained_age(rows)
    with torch.no_grad():
        model.age.fill_(earned)
    return dict(age=earned, rows=rows)


def qualification(age, years):
    """Check an assessed age on the same numeric scale as its stored tensor."""
    import math
    if not math.isfinite(years) or years < 0:
        raise ValueError('required age must be finite and nonnegative')
    if age is None:
        raise ValueError('checkpoint has no assessed age')
    if not torch.is_tensor(age) or age.ndim != 0 or not age.is_floating_point():
        raise ValueError('assessed age must be a floating-point scalar')
    if not math.isfinite(float(age)) or float(age) < 0:
        raise ValueError('assessed age must be finite and nonnegative')
    status = 'not qualified at that age' if bool(age < years) else 'qualified'
    return dict(status=status, required=years, age=float(age))


def checkpoint_qualification(path, task):
    """Check an authenticated native checkpoint against a named row of the age chart."""
    from AgeChart import chart
    ages = {name: age for age, name, _ in chart()}
    if task not in ages:
        raise ValueError(f'unknown age-chart task {task!r}')
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    return qualification(saved['state_dict'].get('age'), ages[task])


def at_age(model, years, measure):
    result = qualification(model.age, years)
    if result['status'] == 'qualified':
        result['result'] = measure()
    return result
