"""Host-side retention of addressed LTM rows at a closed document boundary."""
from dataclasses import dataclass, fields, is_dataclass
import copy
import heapq
import math
import warnings

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ForgettingConfig:
    enabled: bool = True
    high_water: float = .9
    low_water: float = .7
    min_interval: int = 256
    at_epoch: bool = False
    trust_weight: float = 1.
    utility_weight: float = 1.
    luminosity_weight: float = 1.
    age_weight: float = 1.
    half_life: float = 4096.
    similarity: float = .9
    coverage: bool = False
    protect_origins: tuple = (1, 2)

    def __post_init__(self):
        numbers = (self.high_water, self.low_water, self.half_life, self.similarity,
                   self.trust_weight, self.utility_weight, self.luminosity_weight, self.age_weight)
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError('forgetting settings must be finite')
        if not 0 <= self.low_water < self.high_water <= 1:
            raise ValueError('forgetting requires 0 <= low water < high water <= 1')
        if self.min_interval < 0 or self.half_life <= 0 or not 0 <= self.similarity <= 1:
            raise ValueError('invalid forgetting interval, half life or similarity')
        if min(self.trust_weight, self.utility_weight, self.luminosity_weight, self.age_weight) < 0:
            raise ValueError('forgetting weights must be nonnegative')
        if set(self.protect_origins) - {0, 1, 2}:
            raise ValueError('unknown protected LTM origin')

    @classmethod
    def from_xml(cls):
        from util import TheXMLConfig
        names = dict(enabled='forgetting', high_water='forgetHighWater', low_water='forgetLowWater',
            min_interval='forgetMinInterval', at_epoch='forgetAtEpoch', trust_weight='forgetTrustWeight',
            utility_weight='forgetUtilityWeight', luminosity_weight='forgetLuminosityWeight',
            age_weight='forgetAgeWeight', half_life='forgetHalfLife', similarity='forgetSimilarity',
            coverage='forgetCoverage', protect_origins='forgetProtectOrigins')
        defaults = cls()
        values = {}
        for field in fields(cls):
            default = getattr(defaults, field.name)
            value = TheXMLConfig.get('architecture.' + names[field.name], default=default)
            if field.name == 'protect_origins':
                origins = {'conversation': 0, 'provisioned': 1, 'user': 2}
                if isinstance(value, str):
                    try:
                        value = tuple(origins[name] for name in value.split())
                    except KeyError as exc:
                        raise ValueError('unknown protected LTM origin') from exc
                value = tuple(value)
            elif isinstance(default, bool):
                if isinstance(value, str):
                    if value.lower() not in ('true', 'false', '1', '0'):
                        raise ValueError('forgetting boolean must be true or false')
                    value = value.lower() in ('true', '1')
                value = bool(value)
            else:
                value = type(default)(value)
            values[field.name] = value
        return cls(**values)


def history_age(store):
    """Distance from each witnessing stream's tail; the youngest witness wins.

    Re-witnessing does not move a row. Unwitnessed/legacy rows have unknown age
    (zero penalty), never a fabricated global timestamp.
    """
    n = len(store)
    age = store.slots.new_full((n,), torch.inf)
    for stream in range(store.stream.shape[1]):
        rows = store.stream_rows(stream)
        age[rows] = torch.minimum(age[rows], torch.arange(len(rows) - 1, -1, -1,
            device=age.device, dtype=age.dtype))
    return torch.where(torch.isfinite(age), age, 0.)


def _scope_keys(store):
    return [repr((store._semantic_rows.get(int(key), {}).get('bindings', ()),
                  store._semantic_rows.get(int(key), {}).get('scope', ())))
            for key in store.address_keys[:len(store)].tolist()]


def _supports(store):
    """Exact, one-step positive entailments; never infer from negative edges."""
    n = len(store)
    refs = store.refs[:n].tolist()
    positive = ((store.c_plus[:n] > 0) & (store.c_minus[:n] == 0)).tolist()
    trust = store.source_trust[:n].abs().tolist()
    strengths = store.c_plus[:n].tolist()
    scopes = _scope_keys(store)
    by_left = {}
    parts = (store.rel_type[:n] == store.REL_PARTOF).nonzero().flatten().tolist()
    for row in parts:
        if positive[row] and refs[row][0] not in (-1, 0) and refs[row][2] not in (-1, 0):
            by_left.setdefault(refs[row][0], []).append(row)
    result = {}
    for row in parts:
        if not positive[row]:
            continue
        for first in by_left.get(refs[row][0], ()):
            if first == row:
                continue
            for second in by_left.get(refs[first][2], ()):
                if row == second or first == second or refs[second][2] != refs[row][2]:
                    continue
                if not scopes[first] == scopes[second] == scopes[row]:
                    continue
                if min(trust[first], trust[second]) >= trust[row] and min(
                        strengths[first], strengths[second]) >= strengths[row]:
                    result.setdefault(row, []).append(frozenset((first, second)))
    for implication in (store.rel_type[:n] == store.REL_IMPLIES).nonzero().flatten().tolist():
        left, right = (store.index_of_row(refs[implication][role]) for role in (0, 2))
        if left is None or right is None or len({left, right, implication}) < 3:
            continue
        if int(store.rel_type[right]) not in (store.REL_PARTOF, store.REL_IMPLIES):
            continue  # Sentence utility comes from its recorded surprise.
        if not scopes[left] == scopes[right] == scopes[implication]:
            continue
        if all(positive[i] for i in (left, implication, right)) and min(
                trust[left], trust[implication]) >= trust[right] and min(
                strengths[left], strengths[implication]) >= strengths[right]:
            result.setdefault(right, []).append(frozenset((left, implication)))
    return result


def retention_values(store, config, *, alive=None, supports=None):
    """Return independent retention terms; no stored evidence is changed."""
    n = len(store)
    alive = set(range(n)) if alive is None else alive
    supports = _supports(store) if supports is None else supports
    trust = store.source_trust[:n].abs()
    utility = torch.where(store.surprise[:n] < 0, 1., store.surprise[:n]).clamp(0, 1)
    idea = (store.rel_type[:n] == store.REL_NONE) | (store.rel_type[:n] == store.REL_OPERATOR)
    utility = torch.where(idea, utility, 1.)
    for row, choices in supports.items():
        if any(witnesses <= alive for witnesses in choices):
            utility[row] = 0.
    luminosity = torch.ones_like(trust)
    positive, negative = store.c_plus[:n], store.c_minus[:n]
    # Intrinsic both is heterogeneous content, not two disagreeing sources.
    signs = (positive.gt(0) & negative.eq(0)).long() - (negative.gt(0) & positive.eq(0)).long()
    points = F.normalize(store.slots[:n, 0], dim=-1)
    refs = store.refs[:n]
    kinds, orders = store.rel_type[:n], store.order[:n]
    scopes = _scope_keys(store)
    scope_ids = {key: i for i, key in enumerate(dict.fromkeys(scopes))}
    scope = torch.tensor([scope_ids[key] for key in scopes], device=trust.device)
    live = torch.tensor([i in alive for i in range(n)], device=trust.device)
    for start in range(0, n, 128):
        end = min(n, start + 128)
        idea_pair = (kinds[start:end, None] == store.REL_NONE) & (kinds[None] == store.REL_NONE)
        same = torch.where(idea_pair, points[start:end] @ points.T > config.similarity,
            (refs[start:end, None] == refs[None]).all(-1) & (kinds[start:end, None] != store.REL_NONE))
        same &= (kinds[start:end, None] == kinds[None]) & (orders[start:end, None] == orders[None])
        same &= scope[start:end, None] == scope[None]
        contrary = signs[start:end, None] * signs[None] == -1
        weaker = trust[start:end, None] < trust[None]
        luminosity[start:end] = (~(same & contrary & weaker & live[None]).any(-1)).to(trust)
    if config.coverage:
        # Catuskoti coverage over occupied idea coordinates. Opposite poles
        # stay separate; leave-one-out measures the lost positive coverage.
        ids = [i for i in sorted(alive) if int(kinds[i]) == store.REL_NONE]
        if ids:
            codes = store.slots[ids, 0].abs().clamp(0, 1)
            for_pole = codes * positive[ids, None]
            against_pole = codes * negative[ids, None]
            def coverage(p, q):
                if not len(p):
                    return codes.new_zeros(())
                a, b = p.amax(0), q.amax(0)
                return (a - b - torch.minimum(a, b)).mean()
            full = coverage(for_pole, against_pole)
            for local, row in enumerate(ids):
                keep = torch.arange(len(ids), device=codes.device) != local
                luminosity[row] = (full - coverage(for_pole[keep], against_pole[keep])).clamp(0, 1)
    age = history_age(store)
    penalty = -torch.expm1(-age / config.half_life)
    value = (config.trust_weight * trust + config.utility_weight * utility
             + config.luminosity_weight * luminosity - config.age_weight * penalty)
    return dict(value=value, trust=trust, utility=utility, luminosity=luminosity, age=age)


def _references(store, row, context, rows):
    result = set()
    for ref in store._context_occurrence_references(context):
        if ref[:2] == ('ltm', store.address_domain):
            if ref[2] not in rows:
                raise ValueError('forgetting found an unavailable semantic occurrence')
            result.add(rows[ref[2]])
    for ref in store.refs[row].tolist():
        if ref in rows:
            result.add(rows[ref])
    return result


def _trim_expectations(store, alive):
    records = copy.deepcopy(store._expectation_rows)
    for key in list(records):
        if key not in alive:
            del records[key]
            continue
        value = records[key]
        if value['kind'] == 'estimate':
            value['source_occurrences'] = tuple(ref for ref in value['source_occurrences'] if ref[2] in alive)
            target = value['intended_occurrence']
            if target is not None and target[2] not in alive:
                value['intended_occurrence'] = None
        else:
            refs = [ref for ref in (value['estimate_occurrence'], *value.get('previous_estimates', ()))
                    if ref[2] in alive]
            if not refs:
                del records[key]
            else:
                value['estimate_occurrence'] = refs[0]
                value['previous_estimates'] = tuple(refs[1:])
    return records


@torch.no_grad()
def forget(store, config=None, *, document_boundary=False, epoch_boundary=False,
           open_sentence=False, stateless=False, retained_occurrences=()):
    """Plan a complete deletion/coarsening transaction before publishing it."""
    config = config or ForgettingConfig()
    n = len(store)
    if (not config.enabled or stateless or open_sentence or torch.compiler.is_compiling()
            or not (document_boundary or epoch_boundary)):
        return None
    pressure = n >= math.ceil(config.high_water * store.capacity)
    if not pressure and not (epoch_boundary and config.at_epoch):
        return None
    if int(store.forget_passes) and int(store.forget_sentences - store.forget_last_sentence) < config.min_interval:
        return None
    target = math.floor(config.low_water * store.capacity)
    if n <= target:
        return None
    keys = list(map(int, store.address_keys[:n].tolist()))
    indexes = {key: i for i, key in enumerate(keys)}
    contexts = {key: copy.deepcopy(store._semantic_rows.get(key, {})) for key in keys}
    store._validate_constituent_graph(contexts, store.address_domain)
    store._validate_expectation_links()
    if any(store.meaning_of(i) is None for i in range(n)):
        raise ValueError('forgetting requires the complete semantic sidecar')
    dependencies = {i: _references(store, i, contexts[key], indexes) for i, key in enumerate(keys)}
    protected = {i for i in range(n) if int(store.origin[i]) in config.protect_origins}
    for ref in retained_occurrences:
        if ref[:2] != ('ltm', store.address_domain) or ref[2] not in indexes:
            raise ValueError('forgetting live root is unavailable')
        protected.add(indexes[ref[2]])
    pending = list(protected)
    while pending:
        for child in dependencies[pending.pop()]:
            if child not in protected:
                protected.add(child)
                pending.append(child)
    supports = _supports(store)
    terms = retention_values(store, config, supports=supports)
    order = sorted(range(n), key=lambda i: (float(terms['value'][i]), -float(terms['age'][i]), keys[i]))
    alive, advanced, changed, coarsened = set(range(n)), set(), set(), set()
    refs = store.refs[:n].clone()
    store._ensure_texts(n)
    texts = list(store._texts[:n])
    dropped_text = cascade = 0
    cutoff = None
    if len(protected) <= target:
        # A row takes at most one detail stage per pass. Text removal frees
        # sidecar memory, not seats; later passes may coarsen or delete it.
        for row in order:
            if row not in protected and texts[row] is not None:
                texts[row] = None
                advanced.add(row)
                changed.add(row)
                dropped_text += 1

        def deletion(row):
            removed, edits = {row}, {}
            while True:
                previous = set(removed)
                for parent in alive - removed:
                    lost = dependencies[parent] & removed
                    if not lost:
                        continue
                    context = copy.deepcopy(contexts[keys[parent]])
                    roles = list(context.get('role_refs', (None, None, None)))
                    new_refs = refs[parent].clone()
                    coarse = set()
                    for role in range(3):
                        child = indexes.get(int(refs[parent, role]))
                        ref = roles[role]
                        if ref is not None and ref[:2] == ('ltm', store.address_domain):
                            child = indexes[ref[2]]
                        if child in lost and int(store.rel_type[child]) == store.REL_NONE and bool(store.role_mask[parent, role]):
                            roles[role], new_refs[role] = None, -1
                            coarse.add(child)
                    context['role_refs'] = tuple(roles)
                    remaining = {indexes[ref[2]] for ref in store._context_occurrence_references(context)
                                 if ref[:2] == ('ltm', store.address_domain)}
                    if lost - coarse or remaining & lost:
                        removed.add(parent)
                    else:
                        edits[parent] = context, new_refs
                if removed & (protected | advanced):
                    return None
                if removed == previous:
                    break
            edits = {i: value for i, value in edits.items() if i not in removed}
            if set(edits) & (protected | advanced):
                return None
            return removed, edits

        subordinate = set().union(*dependencies.values()) if dependencies else set()
        stages = ([i for i in order if i in subordinate and int(store.record_kind[i]) == store.KINDS.index('unverified')], order)
        for candidates in stages:
            queue = [(float(terms['value'][i]), -float(terms['age'][i]), keys[i], i) for i in candidates]
            heapq.heapify(queue)
            while queue:
                if len(alive) <= target:
                    break
                value, age, key, row = heapq.heappop(queue)
                if row not in alive or row in protected or row in advanced:
                    continue
                if row in supports and not any(proof <= alive - coarsened for proof in supports[row]):
                    utility = (max(0., min(1., float(store.surprise[row])))
                               if int(store.rel_type[row]) == store.REL_NONE and float(store.surprise[row]) >= 0 else 1.)
                    corrected = float(terms['value'][row]) + config.utility_weight * (utility - float(terms['utility'][row]))
                    if corrected > value:
                        heapq.heappush(queue, (corrected, age, key, row))
                        continue
                result = deletion(row)
                if result is None:
                    continue
                removed, edits = result
                # If deducibility made a row expendable, retain a complete
                # proof in the surviving set for the remainder of this pass.
                witnesses = set()
                for item in removed:
                    choices = [proof for proof in supports.get(item, ()) if proof <= alive - coarsened]
                    if choices:
                        available = next((proof for proof in choices if not proof & removed), None)
                        if available is None:
                            break
                        witnesses.update(available)
                else:
                    alive -= removed
                    protected.update(witnesses)
                    cascade += len(removed) - 1
                    cutoff = value
                    for parent, (context, reference) in edits.items():
                        contexts[keys[parent]], refs[parent] = context, reference
                        dependencies[parent] -= removed
                        changed.add(parent)
                        coarsened.add(parent)
                        advanced.add(parent)
                    continue
                # A selected group would remove its own derivation.

    keep = sorted(alive)
    kept_keys = {keys[i] for i in keep}
    contexts = {key: value for key, value in contexts.items() if key in kept_keys}
    store._validate_constituent_graph(contexts, store.address_domain)
    expectations = _trim_expectations(store, kept_keys)
    store._validate_expectation_links(expectations)
    removed = set(range(n)) - alive
    by_origin = {origin: sum(int(store.origin[i]) == origin for i in removed) for origin in (0, 1, 2)}
    store._semantic_rows = contexts
    store._expectation_rows = expectations
    store.refs[:n].copy_(refs)
    store._texts[:n] = texts
    store.compact_rows(keep)
    for row in range(len(store)):
        context = store._semantic_rows.get(int(store.address_keys[row]), {})
        store.metadata_required[row] = bool(context.get('bindings') or context.get('scope') or
            any(ref is not None for ref in context.get('role_refs', ()))) or (
            store.text_of(row) is not None or int(store.address_keys[row]) in expectations
            or int(store.address_keys[row]) in store._definition_rows)
        store._update_semantic_fingerprint(row)
    for old in coarsened & alive:
        row = store.index_of_row(keys[old])
        store._drop_row_postings(row)
        meaning = store.meaning_of(row)
        terms_for_row, complete = store._meaning_leaf_terms(meaning, order=int(store.order[row]))
        store._append_leaf_terms(row, terms_for_row, complete)
    store.forget_passes.add_(1)
    store.forget_last_sentence.copy_(store.forget_sentences)
    store.forget_cutoff.fill_(0. if cutoff is None else cutoff)
    report = dict(before=n, after=len(store), target=target, cutoff=cutoff, deleted_by_origin=by_origin,
                  cascade=cascade, wording=dropped_text, coarsened=len(coarsened),
                  protected=len(protected), target_reached=len(store) <= target)
    store.last_forgetting_report = report
    from util import TheMessage
    TheMessage(f'[LTM forgetting] {report}')
    if not report['target_reached']:
        warnings.warn('LTM forgetting could not reach low water without violating protection or gradual detail loss',
                      RuntimeWarning, stacklevel=2)
    return report


def _live_occurrences(value, namespace):
    pending, found, seen = [value], set(), set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, field.name) for field in fields(item))
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            if (len(item) == 3 and isinstance(item[0], str) and item[0] == 'ltm'
                    and isinstance(item[1], str) and item[1] == namespace and type(item[2]) is int):
                found.add(tuple(item))
            else:
                pending.extend(item)
    return found


def model_boundary(model, *, document_boundary=False, epoch_boundary=False):
    """Training/stateful host hook; live owners provide their actual roots."""
    store = getattr(getattr(model, 'symbolSpace', None), 'ltm_store', None)
    if store is None:
        return None
    roots = _live_occurrences(getattr(model, '_last_understanding', None), store.address_domain)
    memory = getattr(model.symbolSpace, 'what_memory', None)
    if memory is not None:
        roots.update(memory.retained_ltm_occurrences(store.address_domain))
    discourse = getattr(model.symbolSpace, 'expectation', None)
    if discourse is not None:
        for chain in getattr(discourse, '_inter_context_occurrences', ()):
            roots.update(ref for ref in chain if ref is not None)
        for pending in getattr(discourse, '_inter_last_meaning', ()):
            if pending is not None:
                roots.update(getattr(pending, 'source_occurrences', ()))
    result = forget(store, ForgettingConfig.from_xml(), document_boundary=document_boundary,
        epoch_boundary=epoch_boundary, open_sentence=getattr(model, '_open_sentence_slot', None) is not None,
        retained_occurrences=roots)
    if result is not None:
        truth = getattr(model.symbolSpace, 'truth_layer', None)
        if truth is not None:
            truth.sync_from_ltm()
    return result
