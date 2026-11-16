"""Read native sentence trees without retaining any effect of the probe."""
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
                # external engines and file handles are not model state.
                module = type(value).__module__
                if isinstance(value, torch.nn.Module) or module.split('.')[0] in (
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
    for action, (kind, local, word) in enumerate(program.actions.tolist()):
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
                        concepts.append(dict(concept=concept, row=owner._csw_row_of(concept),
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
    """Observe native masks and choices, without choosing or replaying an act."""
    import Attention
    import ModelAttention
    walks, current = [], []
    narrow, mask = ModelAttention.narrow_words, Attention.narrowing_mask
    chooser = model._stm_reducer()
    attend = chooser.attend
    override = vars(chooser).get('attend')
    had_override = 'attend' in vars(chooser)

    def observed_mask(pure, both, neither, **kwargs):
        result = mask(pure, both, neither, **kwargs)
        if current:
            current[-1]['masks'].append(dict(pure=pure.tolist(), both=both.tolist(),
                neither=neither.tolist(), **{k:v.tolist() for k,v in kwargs.items()},
                progress=result.tolist()))
        return result

    def observed_attend(keys, legal, space, **kwargs):
        result = attend(keys, legal, space, **kwargs)
        if current:
            current[-1]['choices'].append(dict(legal=legal.tolist(), action=result[0].tolist(),
                logits=result[3]['logits'].detach().tolist()))
        return result

    def observed_narrow(*args, **kwargs):
        item = dict(masks=[], choices=[], budget=kwargs['budget'], spans=args[2].tolist())
        current.append(item)
        try:
            result = narrow(*args, **kwargs)
            item.update(actions=result.actions.tolist(), accepted=result.accepted.tolist(),
                        descended=result.descended.tolist(), spent=result.table.spent.tolist(),
                        remaining=result.table.remaining.tolist())
            walks.append(item)
            return result
        finally:
            current.pop()
    ModelAttention.narrow_words, Attention.narrowing_mask = observed_narrow, observed_mask
    chooser.attend = observed_attend
    try:
        yield walks
    finally:
        ModelAttention.narrow_words, Attention.narrowing_mask = narrow, mask
        if had_override:
            chooser.attend = override
        else:
            del chooser.attend


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
                evidence = [store.parts.evidence(owner.definitions.row(word, obj)) for obj in objects]
                attention = (None if reading is None else 'minted' if bool(reading.accepted[0, position])
                             else 'entered' if bool(reading.descended[0, position]) else 'left')
                words.append(dict(position=position, form=form, known=before.get(form) is not None,
                    word=word, word_row=None if word is None else owner._csw_row_of(word),
                    objects=list(objects), definition_evidence=[[float(v) for v in pair] for pair in evidence],
                    trusted=bool(evidence) and all(float(pair[0]) >= definition_threshold() for pair in evidence),
                    attention=attention,
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
                selected = [walk for walk in walks if walk['actions'] == reading.actions.tolist()]
                result['attention'] = dict(accepted=reading.accepted[0].tolist(),
                    descended=reading.descended[0].tolist(), actions=reading.actions[0].tolist(),
                    round_words=reading.round_words[0].tolist(), spent=int(reading.table.spent[0]),
                    remaining=int(reading.table.remaining[0]),
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

