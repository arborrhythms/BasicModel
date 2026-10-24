"""Host-side retention of addressed LTM rows at a closed document boundary."""
from dataclasses import dataclass, fields, is_dataclass
import copy
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


def part_supports(parts):
    """Two-hop positive Part proofs; negative testimony never licenses a hop."""
    outgoing, result = {}, {}
    positive = {row for row in parts.indices().tolist()
                if float(parts.truth_plus[row]) > 0 and float(parts.truth_minus[row]) == 0}
    for row in positive:
        outgoing.setdefault(int(parts.refs[row, 0]), []).append(row)
    for row in positive:
        left, right = parts.refs[row].tolist()
        for first in outgoing.get(left, ()):
            for second in outgoing.get(int(parts.refs[first, 1]), ()):
                if len({row, first, second}) != 3 or int(parts.refs[second, 1]) != right:
                    continue
                if min(float(parts.truth_plus[first]), float(parts.truth_plus[second])) >= float(parts.truth_plus[row]):
                    result.setdefault(row, []).append(frozenset((first, second)))
    return result


@torch.no_grad()
def forget_parts(parts, config, *, retained=()):
    """Retention over the same six columns for taxonomy and definitions."""
    if len(parts) < math.ceil(config.high_water * parts.capacity):
        return None
    from RowTime import encoder
    rows = parts.indices().tolist()
    target = math.floor(config.low_water * parts.capacity)
    ticks = encoder().decode_index(parts.when[rows])
    known = parts.when[rows].ne(0).any(-1)
    tail = int(ticks[known].max()) if bool(known.any()) else 0
    ages = {row: max(0, tail-int(ticks[i])) if bool(known[i]) else 0 for i, row in enumerate(rows)}
    proofs = part_supports(parts)
    protected = {int(ref[1]) for ref in retained if ref[:1] == ('sym',) and parts.contains(ref[1])}
    # A referenced edge is released only on the next boundary pass, even
    # when its last parent is selected below. Closure protects every level.
    protected.update(int(child) for row in rows for child in parts.refs[row].tolist()
                     if parts.contains(child))
    pending = list(protected)
    while pending:
        for child in parts.refs[pending.pop()].tolist():
            if parts.contains(child) and child not in protected:
                protected.add(child)
                pending.append(child)
    alive = set(rows)
    def value(row):
        utility = 1. if float(parts.surprise[row]) < 0 else float(parts.surprise[row])
        if any(proof <= alive for proof in proofs.get(row, ())):
            utility = 0.
        access = math.log1p(int(parts.access_count[row]))
        truth = max(float(parts.truth_plus[row]), float(parts.truth_minus[row]))
        return (config.trust_weight*truth + config.utility_weight*utility
                + config.luminosity_weight*access - config.age_weight*(-math.expm1(-ages[row]/config.half_life)))
    for row in sorted(rows, key=lambda row: (value(row), -ages[row], row)):
        if len(alive) <= target:
            break
        if row in protected:
            continue
        available = next((proof for proof in proofs.get(row, ()) if proof <= alive-{row}), None)
        if available is not None:
            protected.update(available)
        alive.remove(row)
    parts.retain(sorted(alive))
    return dict(before=len(rows), after=len(alive), target=target, protected=len(protected),
                target_reached=len(alive) <= target)


def retention_values(store, config, *, alive=None, supports=None):
    """Return independent retention terms; no stored evidence is changed."""
    n = len(store)
    alive = set(range(n)) if alive is None else alive
    supports = {} if supports is None else supports
    trust = store.source_trust[:n].abs()
    utility = torch.where(store.surprise[:n] < 0, 1., store.surprise[:n]).clamp(0, 1)
    for row, choices in supports.items():
        if any(witnesses <= alive for witnesses in choices):
            utility[row] = 0.
    luminosity = torch.ones_like(trust)
    positive, negative = store.c_plus[:n], store.c_minus[:n]
    # Intrinsic both is heterogeneous content, not two disagreeing sources.
    signs = (positive.gt(0) & negative.eq(0)).long() - (negative.gt(0) & positive.eq(0)).long()
    points = F.normalize(store.slots[:n, 0], dim=-1)
    orders = store.order[:n]
    scopes = _scope_keys(store)
    scope_ids = {key: i for i, key in enumerate(dict.fromkeys(scopes))}
    scope = torch.tensor([scope_ids[key] for key in scopes], device=trust.device)
    live = torch.tensor([i in alive for i in range(n)], device=trust.device)
    for start in range(0, n, 128):
        end = min(n, start + 128)
        same = points[start:end] @ points.T > config.similarity
        same &= orders[start:end, None] == orders[None]
        same &= scope[start:end, None] == scope[None]
        contrary = signs[start:end, None] * signs[None] == -1
        weaker = trust[start:end, None] < trust[None]
        luminosity[start:end] = (~(same & contrary & weaker & live[None]).any(-1)).to(trust)
    if config.coverage:
        # Catuskoti coverage over occupied idea coordinates. Opposite poles
        # stay separate; leave-one-out measures the lost positive coverage.
        ids = [i for i in sorted(alive) if int(store.role_mask[i].sum()) == 1]
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
    return result


def _trim_expectations(store, alive):
    records = copy.deepcopy(store._expectation_rows)
    def retained(reference):
        return (store.parts.contains(reference[1]) if reference[0] == 'sym'
                else reference[2] in alive)
    for key in list(records):
        if key not in alive:
            del records[key]
            continue
        value = records[key]
        if value['kind'] == 'estimate':
            value['source_occurrences'] = tuple(ref for ref in value['source_occurrences'] if retained(ref))
            target = value['intended_occurrence']
            if target is not None and not retained(target):
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
    pressure = bool(store.forget_pending) or n >= math.ceil(config.high_water * store.capacity)
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
    # A live segment cannot lose its boundary and acquire the preceding
    # document's cover. Release the empty cover at the next boundary pass.
    from DocumentCover import segments
    for row, covers in segments(store).items():
        for cover in covers:
            boundary = store.location(cover)
            if boundary != row:
                protected.add(boundary)
    for ref in retained_occurrences:
        if ref[:1] == ('sym',) and store.parts.contains(ref[1]):
            continue
        if ref[:2] != ('ltm', store.address_domain) or ref[2] not in indexes:
            raise ValueError('forgetting live root is unavailable')
        protected.add(indexes[ref[2]])
    pending = list(protected)
    while pending:
        for child in dependencies[pending.pop()]:
            if child not in protected:
                protected.add(child)
                pending.append(child)
    supports = {}
    terms = retention_values(store, config, supports=supports)
    order = sorted(range(n), key=lambda i: (float(terms['value'][i]), -float(terms['age'][i]), keys[i]))
    alive, advanced, coarsened = set(range(n)), set(), set()
    store._ensure_texts(n)
    texts = list(store._texts[:n])
    dropped_text = 0
    cutoff = None
    # Freeze incoming references for the entire pass. Dropping a parent's
    # detail cannot make its child eligible at this same boundary.
    parents = {i: set() for i in range(n)}
    for parent, children in dependencies.items():
        for child in children:
            parents[child].add(parent)
    referenced = {row for row, incoming in parents.items() if incoming}
    if len(protected) <= target:
        for row in order:
            if row not in protected and texts[row] is not None:
                texts[row] = None
                advanced.add(row)
                dropped_text += 1

        def advance(row, visiting):
            if row not in alive or row in protected or row in advanced or row in visiting:
                return False
            if row in referenced:
                # Honor the selected child's value by advancing its owners,
                # then stop this pass; do not substitute costlier deletions.
                return any(advance(parent, visiting | {row})
                           for parent in sorted(parents[row], key=order.index))
            context = contexts[keys[row]]
            references = context.get('role_refs', (None, None, None))
            children = dependencies[row]
            if any(ref is not None for ref in references) and all(
                    int(store.role_mask[child].sum()) == 1 for child in children):
                coarse = dict(context, role_refs=(None, None, None))
                remaining = _references(store, row, coarse, indexes)
                if remaining:
                    # A semantic binding still needs its target. Clearing
                    # structural detail must not silently erase that binding.
                    return False
                contexts[keys[row]] = coarse
                coarsened.add(row)
                advanced.add(row)
            else:
                # A bare row is its final stage. An unfused relation cannot
                # keep a child's three-role value as a gist in one argument.
                alive.remove(row)
            return True

        for row in order:
            if len(alive) <= target:
                break
            if row not in alive or row in protected or row in advanced:
                continue
            stage = row in referenced
            if advance(row, set()):
                cutoff = float(terms['value'][row])
                if stage or row in coarsened:
                    break

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
    store._texts[:n] = texts
    store.compact_rows(keep)
    for row in range(len(store)):
        context = store._semantic_rows.get(int(store.address_keys[row]), {})
        store.metadata_required[row] = bool(context.get('bindings') or context.get('scope') or
            any(ref is not None for ref in context.get('role_refs', ()))) or (
            store.text_of(row) is not None or int(store.address_keys[row]) in expectations)
        store._update_semantic_fingerprint(row)
    for old in coarsened & alive:
        row = store.index_of_row(keys[old])
        store._drop_row_postings(row)
        meaning = store.meaning_of(row)
        terms_for_row, complete = store._meaning_leaf_terms(meaning, order=int(store.order[row]))
        store._append_leaf_terms(row, terms_for_row, complete)
    store.forget_passes.add_(1)
    store.forget_pending.fill_(len(store) > target)
    store.forget_last_sentence.copy_(store.forget_sentences)
    store.forget_cutoff.fill_(0. if cutoff is None else cutoff)
    report = dict(before=n, after=len(store), target=target, cutoff=cutoff, deleted_by_origin=by_origin,
                  cascade=0, wording=dropped_text, coarsened=len(coarsened),
                  protected=len(protected | referenced), target_reached=len(store) <= target)
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
    config = ForgettingConfig.from_xml()
    if config.enabled and getattr(model, '_open_sentence_slot', None) is None and (document_boundary or epoch_boundary):
        audit_roots = {ref for value in store._expectation_rows.values()
                      for ref in store._expectation_references(value) if ref[0] == 'sym'}
        def symbols(value):
            if isinstance(value, dict):
                return set().union(*(symbols(item) for item in value.values())) if value else set()
            if isinstance(value, (tuple, list)):
                if len(value) == 2 and value[0] == 'sym' and type(value[1]) is int:
                    return {tuple(value)}
                return set().union(*(symbols(item) for item in value)) if value else set()
            return set()
        # Snapshot all owned structure before either stratum advances. The
        # second half of this boundary cannot release Part rows for the first.
        structure = symbols(store._semantic_rows)
        store.last_part_forgetting_report = forget_parts(store.parts, config,
            retained=roots | audit_roots | structure)
    result = forget(store, config, document_boundary=document_boundary,
        epoch_boundary=epoch_boundary, open_sentence=getattr(model, '_open_sentence_slot', None) is not None,
        retained_occurrences=roots)
    if result is not None:
        truth = getattr(model.symbolSpace, 'truth_layer', None)
        if truth is not None:
            truth.sync_from_ltm()
    return result
