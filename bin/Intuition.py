"""Bounded reference plans for the order-zero thought calculus.

Plans contain addresses and immutable observations, never learned tensors or
LTM parameters. The selected answer reads only its own output-owned sparse weights live.
"""
from dataclasses import dataclass, replace
import torch
from Meaning import ConceptualMeaning
from ThoughtReferences import bindings, with_slots


@dataclass(frozen=True)
class Idea:
    row: int | None = None
    children: tuple = ()
    negative: bool = False
    observed: torch.Tensor | None = None
    evidence: tuple | None = None


@dataclass(frozen=True)
class Term:
    rule: tuple
    antecedent: Idea
    consequent: Idea
    trust: tuple
    meaning: ConceptualMeaning
    knowing: tuple
    premise: tuple
    pi: tuple = ()
    sigma: int | None = None


def pack_idea(plan):
    return (plan.row, tuple(pack_idea(child) for child in plan.children), plan.negative,
            None if plan.observed is None else tuple(plan.observed.detach().tolist()), plan.evidence)


def unpack_idea(record, like):
    row, children, negative, observed, evidence = record
    return Idea(row, tuple(unpack_idea(child, like) for child in children), negative,
        None if observed is None else like.new_tensor(observed), evidence)


def pack_meaning(meaning):
    return (tuple(tuple(row) for row in meaning.roles.detach().tolist()),
        tuple(meaning.role_mask.tolist()), meaning.polarity, meaning.role_refs,
        meaning.bindings, meaning.scope, tuple(pack_meaning(child) for child in meaning.constituents),
        meaning.sentence_kind)


def unpack_meaning(record, like):
    roles, mask, polarity, refs, data, scope, children, kind = record
    return ConceptualMeaning(like.new_tensor(roles), torch.tensor(mask, device=like.device),
        polarity, refs, data, scope, tuple(unpack_meaning(child, like) for child in children), kind)


def pack_term(term):
    return (term.rule, pack_idea(term.antecedent), pack_idea(term.consequent), term.trust,
            pack_meaning(term.meaning), tuple(pack_idea(plan) for plan in term.knowing), term.premise, term.pi, term.sigma)


def unpack_term(record, like):
    rule, a, b, trust, meaning, knowing, premise, pi, sigma = record
    return Term(rule, unpack_idea(a, like), unpack_idea(b, like), trust,
        unpack_meaning(meaning, like), tuple(unpack_idea(plan, like) for plan in knowing), premise, pi, sigma)


def held_terms(meaning):
    return tuple(unpack_term(record, meaning.roles) for record in bindings(meaning).get('_intuition_terms', ()))


def observation(owner, value):
    """Keep only order-zero content in an immutable working-value plan."""
    return Idea(observed=encode(owner, chart(owner, value.detach())).detach().clone())


def _without_concept(owner, ref):
    from LTM1 import NULL_CONCEPT
    if ref is None or ref[0] not in ('sym', 'ltm', 'name'):
        return False
    if ref == ('sym', NULL_CONCEPT):
        return True
    key = owner.mtm.token_of(ref[1] if ref[0] == 'sym' else ref)
    return key is not None and owner.mtm.entries[key].kind == NULL_CONCEPT


def extract(owner, meaning, *, work, seen=()):
    """Expand higher orders; an individual's working value enters at order 0."""
    work.require('concept')
    alloc = owner._concept_allocator
    layer = alloc.layer(0)
    plans = []
    for slot in (0, 2) if int(meaning.role_mask.sum()) > 1 else (0,):
        if not bool(meaning.role_mask[slot]):
            continue
        ref = meaning.role_refs[slot]
        if ref is not None and ref[0] == 'constituent':
            plans.append(extract(owner, meaning.constituents[ref[1]], work=work, seen=seen))
            continue
        if _without_concept(owner, ref):
            continue
        if ref is not None and ref[0] == 'sym':
            if ref in seen:
                raise ValueError('cyclic intuition definition')
            row = owner._csw_row_of(ref[1])
            if row is None:
                # MTM identities have reserved transient ids beyond the
                # inventory. Their held working content is already order zero;
                # the id must not be looked up as a persistent concept row.
                mtm = owner.mtm
                token = mtm.token_of(ref[1])
                if token is not None and mtm.entries[token].transient == ref[1]:
                    work.require('payload')
                    plans.append(observation(owner, meaning.roles[slot]))
                    continue
                if alloc.parts.contains(ref[1]):
                    work.require('record')
                    store = owner._closed_clause_store()
                    if store is None:
                        raise ValueError('intuition has no owner for its referenced relation')
                    relation = store.part_row(ref[1], reading='part')['meaning']
                    plans.append(extract(owner, relation, work=work, seen=(*seen, ref)))
                    continue
                raise ValueError('intuition references an unallocated concept: '+repr(ref))
            if alloc.order_of(ref[1]) == 0:
                plans.append(Idea(row=row))
            else:
                children = []
                for child_ref in alloc.refs(ref[1], 'part'):
                    if not isinstance(child_ref, tuple) or child_ref[0] not in ('sym', 'field'):
                        continue
                    child = replace(ConceptualMeaning.from_description(meaning.roles[slot]),
                                    role_refs=(('sym', child_ref[1]), None, None))
                    children.append(extract(owner, child, work=work, seen=(*seen, ref)))
                if not children:
                    # An individual's working value is its current order-zero
                    # constituents. Its identity code is never looked up here.
                    children.append(observation(owner, meaning.roles[slot]))
                plans.append(Idea(children=tuple(children)))
        else:
            plans.append(observation(owner, meaning.roles[slot]))
    return Idea(children=tuple(plans), negative=not meaning.polarity)


def chart(owner, value):
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    form = derived.percept_event_width if derived is not None else value.shape[-1]
    value = value.clamp(0, 1)
    if derived is not None:
        value = torch.cat((value[:derived.percept_width],
                           torch.zeros_like(value[derived.percept_width:form]), value[form:]))
    positive, negative = value[form:].chunk(2) if form < len(value) else (value[:0], value[:0])
    return torch.stack((torch.cat((value[:form], positive)),
                        torch.cat((torch.zeros_like(value[:form]), negative))), -1)


def encode(owner, field):
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    form = derived.percept_event_width if derived is not None else len(field)
    # Form identifies the event type in either stance. Only the complement
    # stores its two poles; the complete paired field remains in the proof.
    return torch.cat((field[:form].amax(-1), field[form:, 0], field[form:, 1]))


@torch.no_grad()
def leaf(owner, row):
    book = owner.similarity_codebook
    return book.lookup_rows(torch.tensor(row, device=book.W.device)).detach().clamp(0, 1)


@torch.no_grad()
def read(owner, plan):
    values = [read(owner, child) for child in plan.children]
    if plan.row is not None:
        values.append(chart(owner, leaf(owner, plan.row)))
    if plan.observed is not None:
        values.append(chart(owner, plan.observed.detach()))
    base = torch.stack(values).amax(0) if values else chart(owner,
        owner.similarity_codebook.W.new_zeros(owner.outputShape[-1]))
    base = base.flip(-1) if plan.negative else base
    if plan.evidence is not None:
        base = torch.maximum(torch.minimum(base, base.new_tensor(plan.evidence[0])),
                             torch.minimum(base.flip(-1), base.new_tensor(plan.evidence[1])))
    return base


def literals(plan, negative=False):
    negative = negative ^ plan.negative
    if plan.children and plan.row is None and plan.observed is None and plan.evidence is None:
        return tuple(literal for child in plan.children for literal in literals(child, negative))
    return (replace(plan, negative=negative),)


def seed(owner, term, work):
    leaves = literals(term.antecedent)
    for _ in range(len(leaves) + 1):
        work.require('edge')
    # Reference paths, not values, identify persistent weights. Observed
    # literals use their structural position within the remembered rule.
    keys = tuple((i, leaf.row, leaf.negative) for i, leaf in enumerate(leaves))
    pi, sigma = owner.intuition.seed(term.rule, keys)
    return replace(term, pi=pi, sigma=sigma)


def _terms(owner, term):
    known = torch.stack([read(owner, plan) for plan in term.knowing]).amax(0)
    values = torch.stack([read(owner, plan) for plan in literals(term.antecedent)])
    antecedent = values.amax(0)
    return known, values, antecedent, read(owner, term.consequent)


def _evaluate(known, literals, consequence, trust, pi, sigma):
    from IntuitionLayer import power
    antecedent = literals.amax(0)
    meets = torch.minimum(known[None], literals)
    numerator = power(meets, pi[:, None, None]).amax(0)
    coverage = (numerator.sum() / antecedent.sum().clamp_min(1e-12)).clamp(0, 1)
    firing = torch.minimum(coverage, coverage.new_tensor(trust))
    field = torch.maximum(torch.minimum(consequence, firing[0]),
                          torch.minimum(consequence.flip(-1), firing[1]))
    return power(field, sigma, sigma=True), firing


def evaluate(owner, term):
    known, values, antecedent, consequence = _terms(owner, term)
    bank = getattr(owner, 'intuition', None)
    pi = (values.new_ones(len(values)) if not term.pi else bank.weight('pi', term.pi))
    sigma = values.new_ones(()) if term.sigma is None else bank.weight('sigma', term.sigma)
    return _evaluate(known, values, consequence, term.trust, pi, sigma)


def evidence_pair(owner, term):
    known, values, antecedent, _ = _terms(owner, term)
    coverage = torch.minimum(known, antecedent).sum() / antecedent.sum().clamp_min(1e-12)
    return torch.minimum(coverage, coverage.new_tensor(term.trust)).detach()


@torch.no_grad()
def proof(owner, term, *, accepted=False):
    known, values, antecedent, consequence = _terms(owner, term)
    pi = values.new_ones(len(values)) if not term.pi else owner.intuition.weight('pi', term.pi)
    sigma = values.new_ones(()) if term.sigma is None else owner.intuition.weight('sigma', term.sigma)
    return dict(rule=term.rule, trust=term.trust, knowing=known, literals=values,
        consequent=consequence, pi=pi.clone(), sigma=sigma.clone(), accepted=accepted)


@torch.no_grad()
def execute(owner, meaning, frames, *, row, work, premise):
    meaning = ConceptualMeaning.from_description(meaning)
    implicit = pervasion(owner, meaning, frames=frames, work=work)
    if implicit is not None:
        return implicit
    known = [extract(owner, meaning, work=work)]
    # Values in the prepended field are the actual working values, including
    # transient candidates with no persistent concept row.
    model = getattr(owner, '_model', None)
    entries = getattr(getattr(model, 'inputSpace', None), 'symbol_entries', None)
    if entries is not None:
        for column in range(entries.prepend):
            identity = int(entries.concept_ids[row, column])
            if identity == -1 or _without_concept(owner, ('sym',identity)):
                continue
            work.require('payload')
            known.append(observation(owner, entries.lookup_atoms[row, column]))
    expected = getattr(owner, '_observed_knowing', None)
    knowing_stop = work.spent + work.remaining // 3
    if expected is not None:
        for address in (expected[row].amax(-1) > 0).nonzero().flatten().tolist():
            if work.spent + 2 > knowing_stop:
                break
            cid = owner.concept_id_at_row(address)
            if cid is not None:
                value = replace(ConceptualMeaning.from_description(owner.similarity_codebook.W[address].detach()),
                                role_refs=(('sym', cid), None, None))
                known.append(replace(extract(owner, value, work=work),
                                     evidence=tuple(expected[row, address].tolist())))
    terms, fields, pairs, proofs = [], [], [], []
    for frame in frames:
        rule = frame['meaning']
        left = operand(rule, 0)
        right = operand(rule, 2)
        from QueryWork import QueryWorkExhausted
        try:
            a = extract(owner, left, work=work)
            b = extract(owner, right, work=work)
        except QueryWorkExhausted:
            break
        term = Term(frame['occurrence'], a, b, tuple(frame['evidence']), right,
                    tuple(known), premise)
        try:
            term = seed(owner, term, work)
        except QueryWorkExhausted:
            break
        field, firing = evaluate(owner, term)
        if bool(firing.any()):
            terms.append(term)
            fields.append(field)
            pairs.append(firing)
            proofs.append(proof(owner, term))
    field = torch.stack(fields).amax(0) if fields else read(owner, Idea())
    pair = torch.stack(pairs).amax(0) if pairs else field.new_zeros(2)
    def leaves(plan):
        if plan.row is not None:
            yield plan.row
        for child in plan.children:
            yield from leaves(child)
    knowing = []
    for address in dict.fromkeys(address for term in terms for address in leaves(term.consequent)):
        if not work.consume('effect_node'):
            break
        code = chart(owner, leaf(owner, address)).amax(-1)
        poles = torch.minimum(field, code[:, None]).sum(0) / code.sum().clamp_min(1e-12)
        knowing.append((address, tuple(poles.tolist())))
    value = encode(owner, field).detach()
    result = with_slots(replace(ConceptualMeaning.from_description(value), bindings=dict(
        _intuition_terms=tuple(pack_term(term) for term in terms),
        _intuition_firings=tuple(tuple(map(float, pair)) for pair in pairs),
        _intuition_field=tuple(tuple(row) for row in field.detach().tolist()),
        _evidence_kind='intuition')), (), pair=pair)
    return dict(value=value, meaning=result, support_true=float(pair[0]), support_false=float(pair[1]),
        witnesses=tuple(term.rule for term in terms), sources=tuple(term.rule for term in terms),
        firings=tuple(tuple(map(float, pair)) for pair in pairs), evidence_kind='intuition',
        intuition_derivation=tuple(proofs), knowing=tuple(knowing), incomplete=())


def replay(proofs):
    fields, pairs = [], []
    for proof in proofs:
        field, pair = _evaluate(proof['knowing'], proof['literals'], proof['consequent'],
                                proof['trust'], proof['pi'], proof['sigma'])
        if proof['accepted']:
            antecedent = proof['literals'].amax(0)
            coverage = torch.minimum(proof['knowing'], antecedent).sum() / antecedent.sum().clamp_min(1e-12)
            pair = torch.minimum(coverage, coverage.new_tensor(proof['trust']))
        fields.append(field)
        pairs.append(pair)
    if not fields:
        raise ValueError('an empty intuition has no replayed firing')
    return torch.stack(fields).amax(0), torch.stack(pairs).amax(0)


@torch.no_grad()
def write_knowing(model, evidence, row):
    """Deposit paired containments into the existing order-zero field."""
    space = model._concept_owner()
    carrier = getattr(model.conceptualSpace, 'subspace', None)
    existing = getattr(carrier, '_concept_activations', None)
    if existing is None or not evidence.get('knowing'):
        return
    addresses = getattr(carrier, '_concept_inventory_rows', None)
    if addresses is None:
        addresses = space._field_inventory_rows(existing.device)
    addresses = addresses[:, row] if addresses.ndim == 2 else addresses
    field = existing.detach().clone()
    location = getattr(carrier, '_thought_occurrence', None)
    location = 0 if location is None else int(location)
    if not 0 <= location < field.shape[2]:
        raise RuntimeError('intuition has no constructed knowing extent')
    for address, pair in evidence['knowing']:
        slots = (addresses[:len(field)] == address).nonzero().flatten()
        if len(slots) > 1:
            raise ValueError('intuition knowing has duplicate inventory addresses')
        if len(slots):
            index = int(slots[0])
            field[index, row, location] = torch.maximum(field[index, row, location], field.new_tensor(pair))
    object.__setattr__(carrier, '_concept_activations', field)


def operand(rule, slot):
    ref = rule.role_refs[slot]
    if ref is not None and ref[0] == 'constituent':
        return rule.constituents[ref[1]]
    return replace(ConceptualMeaning.from_description(rule.roles[slot]), role_refs=(ref, None, None))


def selected(owner, term):
    field, proposal = evaluate(owner, term)
    pair = evidence_pair(owner, term)
    value = encode(owner, field)
    roles = term.meaning.roles.clone()
    occupied = term.meaning.role_mask.nonzero().flatten().tolist()
    if len(occupied) == 1:
        roles[occupied[0]] = value
    else:
        for slot, plan in zip((0, 2), term.consequent.children):
            content, _ = evaluate(owner, replace(term, consequent=plan))
            roles[slot] = encode(owner, content)
    meaning = replace(term.meaning, roles=roles,
        bindings=dict(bindings(term.meaning), _intuition_selected=pack_term(term),
                      _producing_operation='infer', _evidence_kind='inference',
                      _bound_roles=(('referent', occupied[0] if len(occupied) == 1 else 2),),
                      _thought_witnesses=(term.rule,)))
    return with_slots(meaning, (), pair=pair.detach()), pair


def live_meaning(model, meaning):
    term = bindings(meaning).get('_intuition_selected')
    return meaning if term is None else selected(model._concept_owner(), unpack_term(term, meaning.roles))[0]


def pervade(parts, property_field):
    """Every part must carry the property on each independently reduced lane."""
    if parts.ndim != 3 or parts.shape[1:] != property_field.shape or not len(parts):
        raise ValueError('pervasion requires nonempty paired parts in one order-zero chart')
    code = property_field.amax(-1)
    covered = torch.minimum(parts, code[None, :, None]).sum(1)
    pairs = covered / code.sum().clamp_min(1e-12)
    return pairs.amin(0)


def pervasion(owner, meaning, *, frames=(), work):
    subject, prop = meaning.role_refs[0], meaning.role_refs[2]
    if _without_concept(owner, subject) or _without_concept(owner, prop):
        return None
    if (subject is None or prop is None or subject[0] != 'sym'
            or int(meaning.role_mask.sum()) != 3):
        return None
    from ClauseRow import predicate_identity
    from Queries import GrammaticalThoughtRegistry, THOUGHT_EXECUTORS
    # Pervasion answers the Part/property relation, including its registered
    # taxonomy question. Three roles alone do not give another predicate
    # that meaning or proof.
    descriptor = THOUGHT_EXECUTORS['isPart']
    name = GrammaticalThoughtRegistry._name((descriptor.domain, descriptor.semantic_id))
    taxonomy = getattr(owner, '_frozen_named', {}).get(name)
    predicate = meaning.role_refs[1]
    if (predicate != ('sym', predicate_identity('part'))
            and (taxonomy is None or predicate != ('sym', taxonomy))):
        return None
    alloc = owner._concept_allocator
    from LTM1 import SYMBOL
    from Layers import Ops
    from QueryWork import QueryWorkExhausted
    parts = alloc.parts
    refs = {}
    for kind, cid in parts.parts((SYMBOL, subject[1])):
        if kind == SYMBOL and owner._csw_row_of(cid) is not None:
            edge = parts.find((kind, cid), (SYMBOL, subject[1]))
            refs[('sym', cid)] = tuple(float(pole) for pole in parts.evidence(edge))
    for ref in alloc.refs(subject[1], 'part'):
        if isinstance(ref, tuple) and ref[0] in ('sym', 'field'):
            refs.setdefault(('sym', ref[1]), (float(ref[2]) if len(ref) > 2 else 1., 0.))
    if not refs:
        return None
    property_field = read(owner, extract(owner, operand(meaning, 2), work=work))
    whole = chart(owner, meaning.roles[0])
    surrounding = []
    for kind, cid in parts.wholes((SYMBOL, subject[1])):
        if kind != SYMBOL or owner._csw_row_of(cid) is None:
            continue
        work.require('part')
        value = chart(owner, leaf(owner, owner._csw_row_of(cid)))
        if bool(Ops.partOf(whole.amax(-1), value.amax(-1))):
            edge = parts.find((SYMBOL, subject[1]), (kind, cid))
            trust = value.new_tensor(tuple(float(pole) for pole in parts.evidence(edge)))
            surrounding.append(torch.maximum(torch.minimum(value, trust[0]),
                                              torch.minimum(value.flip(-1), trust[1])))
    fields, witnesses = [], []
    for ref, trust in refs.items():
        work.require('part')
        point = replace(ConceptualMeaning.from_description(meaning.roles[0]), role_refs=(ref, None, None))
        plan = extract(owner, point, work=work)
        raw = read(owner, plan)
        if not bool(Ops.partOf(raw.amax(-1), whole.amax(-1))):
            continue
        value = read(owner, replace(plan, evidence=trust))
        context = torch.stack((value, *surrounding)).amax(0)
        # Rule structure is narrowed to this part. The evidence calculation
        # uses unit exponents: learned proposal strength never becomes truth.
        contributions = [value]
        for frame in frames:
            try:
                work.require('record')
                antecedent = read(owner, extract(owner, operand(frame['meaning'], 0), work=work))
                consequence = read(owner, extract(owner, operand(frame['meaning'], 2), work=work))
            except QueryWorkExhausted:
                break
            coverage = torch.minimum(context, antecedent).sum() / antecedent.sum().clamp_min(1e-12)
            firing = torch.minimum(coverage, coverage.new_tensor(frame['evidence']))
            contributions.append(torch.maximum(torch.minimum(consequence, firing[0]),
                                                torch.minimum(consequence.flip(-1), firing[1])))
        fields.append(torch.stack(contributions).amax(0))
        witnesses.append(ref)
    if not fields:
        return None
    narrowed = torch.stack(fields)
    pair = pervade(narrowed, property_field)
    result = with_slots(replace(meaning, bindings=dict(bindings(meaning),
        _evidence_kind='pervasion', _pervasion_subject=subject,
        _pervasion_property=prop, _thought_witnesses=tuple(witnesses))), (), pair=pair)
    return dict(value=result.roles[0], meaning=result, support_true=float(pair[0]),
        support_false=float(pair[1]), witnesses=tuple(witnesses), sources=tuple(witnesses),
        evidence_kind='pervasion', reason='part_minimum', incomplete=(),
        pervasion_derivation=dict(parts=narrowed.detach(), property=property_field.detach()))


def judge(meaning, records):
    """Union independent implicit and explicit proofs at the same close."""
    from ThoughtReferences import evidence_pair
    from ThoughtAnswer import accepted
    if not accepted(meaning):
        return meaning
    pair = evidence_pair(meaning)
    judged = []
    for record in records:
        result = getattr(record, 'result', None)
        other = record.meaning if result is None else result.evidence.get('meaning', record.meaning)
        if (other is None or bindings(other).get('_evidence_kind') != 'pervasion'
                or other.role_refs != meaning.role_refs):
            continue
        implicit = evidence_pair(other)
        pair = tuple(max(a, b) for a, b in zip(pair, implicit))
        judged.append(implicit)
    if not judged:
        return meaning
    return with_slots(replace(meaning, bindings=dict(bindings(meaning),
        _implicit_judgements=tuple(judged))), (), pair=pair)
