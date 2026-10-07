"""Grammar-derived, inverted retrieval index over completed LTM fields.

Only code-to-row postings persist. No forward list of a sentence's words,
leaves, operations, or activation is retained with a row.
"""
from __future__ import annotations

import heapq
import itertools
import math

import torch
from torch.nn import functional as F


class LeafCodeIndex:
    """The row writer owns one derived inverted index, without forward lists."""

    def _init_leaf_index(self):
        self.register_buffer('leaf_complete', torch.zeros(self.capacity, 3, dtype=torch.bool))
        self.register_buffer('index_stream', torch.full((self.capacity,), -1, dtype=torch.long))
        # These columns serialize global postings, never per-row sequences.
        for name in ('posting_codes', 'posting_roles', 'posting_rows'):
            self.register_buffer(name, torch.empty(0, dtype=torch.long))
        self._leaf_postings = {}
        self._index_occurrences = {}
        self._index_code_row = None
        self._index_unfold = None
        self._index_order_of = None
        self._leaf_needs_owner_reindex = False

    def configure_leaf_index(self, *, code_row=None, unfold=None, order_of=None):
        """Bind the current grammar owner. Code addresses do not supply terms."""
        if any(value is not None and not callable(value) for value in (code_row, unfold, order_of)):
            raise TypeError('leaf index adapters must be callable')
        self._index_code_row, self._index_unfold = code_row, unfold
        self._index_order_of = order_of

    @staticmethod
    def _checked_leaf_terms(terms):
        terms = tuple(tuple(values) for values in terms)
        if len(terms) != 3 or any(type(code) is not int or code < 0
                                  for values in terms for code in values):
            raise ValueError('leaf codes require three sequences of nonnegative code addresses')
        return terms

    def _meaning_leaf_terms(self, meaning, *, order=0, max_nodes=1024, work=None):
        """Unfold the actual occupied values, following only structural refs."""
        remaining, active = [int(max_nodes)], set()
        def visit(value, field_order):
            if id(value) in active or remaining[0] <= 0:
                return ((), (), ()), (False, False, False)
            remaining[0] -= 1
            active.add(id(value))
            terms, complete = [], []
            for role, reference in enumerate(value.role_refs):
                codes, known = (), not bool(value.role_mask[role])
                child = None
                child_order = field_order
                if reference is not None and reference[0] == 'constituent':
                    child = value.constituents[reference[1]]
                elif reference is not None and reference[0] == 'ltm':
                    index = self._index_occurrences.get(reference)
                    child = None if index is None else self.meaning_of(index)
                    child_order = -1 if index is None else int(self.order[index])
                if not known and child is not None:
                    recovered, status = visit(child, child_order)
                    codes, known = tuple(itertools.chain.from_iterable(recovered)), all(status)
                elif not known and field_order >= 0 and self._index_unfold is not None and remaining[0] > 0:
                    allowance = min(remaining[0], work.remaining) if work is not None else remaining[0]
                    recovered = self._index_unfold(value.roles[role].detach(), allowance,
                                                   order=field_order, work=work)
                    if recovered is not None:
                        codes, spent, known = recovered
                        if type(spent) is not int or not 0 <= spent <= remaining[0]:
                            raise ValueError('unfold exceeded its node allowance')
                        remaining[0] -= spent
                        codes = tuple(codes)
                terms.append(codes)
                complete.append(known)
            active.remove(id(value))
            return self._checked_leaf_terms(terms), tuple(complete)
        return visit(meaning, order)

    def leaf_terms(self, row, role):
        """Audit a row by inverting the global postings; no forward list exists."""
        row, role = int(row), int(role)
        if not 0 <= row < len(self) or role not in (0, 1, 2):
            raise IndexError('leaf index row/role is unavailable')
        return tuple(sorted(code for (code, slot), rows in self._leaf_postings.items()
                            if slot == role and row in rows))

    def rows_for_code(self, code, *, role=None):
        """Exact audit view; bounded readers use the posting iterators below."""
        if type(code) is not int or code < 0 or role not in (None, 0, 1, 2):
            raise ValueError('invalid leaf-code cue')
        if role is not None:
            return tuple(self._leaf_postings.get((code, role), ()))
        return tuple(sorted(set(itertools.chain.from_iterable(
            self._leaf_postings.get((code, r), ()) for r in range(3)))))

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        records = [(code, role, row) for (code, role), rows in sorted(self._leaf_postings.items())
                   for row in rows]
        for i, name in enumerate(('posting_codes', 'posting_roles', 'posting_rows')):
            destination[prefix + name] = self.index_stream.new_tensor([record[i] for record in records])

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        legacy = any(prefix + name in state_dict for name in ('leaf_codes', 'leaf_offsets'))
        if legacy:
            import warnings
            warnings.warn('Dropping per-sentence leaf lists; retrieval terms require grammar unfolding (§11.1).',
                          UserWarning, stacklevel=2)
            for name in ('leaf_codes', 'leaf_offsets'):
                state_dict.pop(prefix + name, None)
        names = ('posting_codes', 'posting_roles', 'posting_rows')
        present = [prefix + name in state_dict for name in names]
        if any(present) and not all(present):
            error_msgs.append(prefix + 'incomplete inverted retrieval index')
            return
        self._leaf_index_missing = not all(present) or legacy
        self._leaf_needs_owner_reindex = self._leaf_index_missing
        for name in names:
            value = state_dict.setdefault(prefix + name, self.index_stream.new_empty(0))
            if value.ndim != 1 or value.dtype != torch.long:
                error_msgs.append(prefix + 'invalid inverted retrieval column ' + name)
                return
            setattr(self, name, self.index_stream.new_empty(value.shape))
        state_dict.setdefault(prefix + 'leaf_complete', torch.zeros_like(self.leaf_complete))
        state_dict.setdefault(prefix + 'index_stream', torch.full_like(self.index_stream, -2))
        if legacy:
            state_dict[prefix + 'leaf_complete'] = torch.zeros_like(self.leaf_complete)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if len({getattr(self, name).numel() for name in names}) != 1:
            error_msgs.append(prefix + 'inverted retrieval columns have different lengths')
            return
        self._leaf_postings = {}
        for code, role, row in zip(*(getattr(self, name).tolist() for name in names)):
            if code < 0 or role not in (0, 1, 2) or not 0 <= row < len(self):
                error_msgs.append(prefix + 'invalid inverted retrieval posting')
                return
            rows = self._leaf_postings.setdefault((code, role), [])
            if rows and rows[-1] >= row:
                error_msgs.append(prefix + 'inverted retrieval postings must be unique and sorted')
                return
            rows.append(row)
        for name in names:
            setattr(self, name, self.index_stream.new_empty(0))
        self.rebuild_leaf_postings()

    @torch.no_grad()
    def _append_leaf_terms(self, row, terms, complete, stream=-1):
        terms = self._checked_leaf_terms(terms)
        if self._index_unfold is None and not all(complete):
            self._leaf_needs_owner_reindex = True
        for role, codes in enumerate(terms):
            self.leaf_complete[row, role] = bool(complete[role])
            for code in set(codes):
                rows = self._leaf_postings.setdefault((code, role), [])
                if row not in rows:
                    rows.append(row)
                    rows.sort()
        self.index_stream[row] = int(stream)
        self._index_occurrences[self.occurrence_of(row)] = row

    def rebuild_leaf_postings(self):
        self._index_occurrences = {self.occurrence_of(row): row for row in range(len(self))}
        self._leaf_postings = {key: sorted(set(row for row in rows if 0 <= row < len(self)))
                               for key, rows in self._leaf_postings.items()}

    @torch.no_grad()
    def reindex_meanings(self):
        """Regenerate terms from stored fields with the current grammar owner."""
        streams = self.index_stream.clone()
        self.leaf_complete.zero_()
        self._leaf_postings = {}
        self.rebuild_leaf_postings()
        for row in range(len(self)):
            meaning = self.meaning_of(row)
            terms, complete = (((), (), ()), (False, False, False)) if meaning is None else self._meaning_leaf_terms(meaning, order=int(self.order[row]))
            self._append_leaf_terms(row, terms, complete, int(streams[row]))
        self._leaf_needs_owner_reindex = self._index_unfold is None

    @torch.no_grad()
    def compact_leaf_rows(self, keep):
        """Apply the store's row permutation to the inverted postings."""
        mapping = {int(old): new for new, old in enumerate(keep)}
        self._leaf_postings = {key: sorted(mapping[row] for row in rows if row in mapping)
                               for key, rows in self._leaf_postings.items()}

    @torch.no_grad()
    def remap_leaf_codes(self, mapping):
        """Apply a codebook compaction map without consulting input words."""
        remapped = {}
        for (code, role), rows in self._leaf_postings.items():
            new = mapping.get(code, -1)
            if new < 0:
                self.leaf_complete[rows, role] = False
            else:
                remapped.setdefault((int(new), role), set()).update(rows)
        self._leaf_postings = {key: sorted(rows) for key, rows in remapped.items()}

    @staticmethod
    def _scope_contains(query, stored):
        query, stored = dict(query), dict(stored)
        for name, requested in query.items():
            actual = stored.get(name)
            if (name in ('where', 'when') and isinstance(requested, tuple) and len(requested) == 2
                    and all(isinstance(x, (int, float)) for x in requested)):
                if not isinstance(actual, tuple) or len(actual) != 2:
                    return False
                if not all(isinstance(x, (int, float)) and math.isfinite(x)
                           for x in (*requested, *actual)):
                    return False
                if not requested[0] <= actual[0] <= actual[1] <= requested[1]:
                    return False
            elif actual != requested:
                return False
        return True

    def cued_rows(self, cue, *, primed=(), references=(), retrieved=(), max_candidates=32,
                  work=None, stream=None):
        """Rank K indexed candidates on bound roles, with hard scope isolation.

        Priming widens the cue codes. Contiguity widens only from already
        retrieved rows, in their own stream. No recent-store slice is read.
        Every examined candidate is charged before its payload is accessed.
        """
        if type(max_candidates) is not int or max_candidates < 0:
            raise ValueError('candidate limit must be a nonnegative integer')
        # An already structured query carries native references, whose orders
        # are owned by the concept table. Stored rows always use their stamp.
        cue_order = 0
        if self._index_order_of is not None and self._index_code_row is not None:
            for reference in cue.role_refs:
                if reference is not None and reference[0] in ('sym', 'meta'):
                    code = self._index_code_row(reference)
                    if code is not None:
                        cue_order = max(cue_order, self._index_order_of(code))
        terms, _ = self._meaning_leaf_terms(cue, order=cue_order, work=work)
        postings = [self._leaf_postings.get((code, role), ())
                    for role, codes in enumerate(terms) if bool(cue.role_mask[role])
                    for code in set(codes)]
        postings.extend(self._leaf_postings.get((int(code), role), ())
                        for code in primed for role in range(3))
        explicit = [self._index_occurrences[ref] for ref in references
                    if ref in self._index_occurrences]
        adjacent = set()
        for reference in retrieved:
            row = self._index_occurrences.get(reference)
            if row is None:
                continue
            for neighbor in (row - 1, row + 1):
                if (0 <= neighbor < len(self)
                        and int(self.index_stream[row]) >= 0
                        and int(self.index_stream[neighbor]) == int(self.index_stream[row])):
                    adjacent.add(neighbor)
        postings.extend((sorted(explicit), sorted(adjacent)))
        candidates = heapq.merge(*(iter(rows) for rows in postings))
        found, seen, scanned, incomplete = [], set(), 0, []
        for row in candidates:
            if row in seen:
                continue
            seen.add(row)
            if scanned >= max_candidates:
                incomplete.append('candidate_limit')
                break
            if work is not None and not work.consume('record'):
                incomplete.append('work_budget')
                break
            scanned += 1
            if stream is not None and int(self.index_stream[row]) not in (-1, int(stream)):
                continue
            if self.KINDS[int(self.record_kind[row])] in ('estimate', 'question'):
                continue
            meaning = self.meaning_of(row)
            if meaning is None:
                incomplete.append('unavailable_metadata')
                continue
            if not self._scope_contains(cue.scope, meaning.scope):
                continue
            if cue.bindings and not self._scope_contains(cue.bindings, meaning.bindings):
                continue
            mask = cue.role_mask.to(device=meaning.roles.device)
            if bool((mask & ~meaning.role_mask).any()):
                continue
            similarities = F.cosine_similarity(cue.roles.detach().to(meaning.roles), meaning.roles, dim=-1)
            match = float(similarities[mask].clamp(0, 1).mean()) if bool(mask.any()) else 0.
            record = self.row(row)
            found.append(dict(record, index=row, match=match, contiguous=row in adjacent,
                              leaf_codes=tuple(self.leaf_terms(row, role) for role in range(3))))
        found.sort(key=lambda item: (-item['match'], -int(item['contiguous']), item['index']))
        return {'value': tuple(found), 'records_scanned': scanned,
                'incomplete': tuple(dict.fromkeys(incomplete))}


@torch.no_grad()
def unfold_idea(language, basis, idea, limit, *, work=None, activation=None,
                order=0, order_of=None, sigma_inverse=None):
    """Unfold a detached idea with the current generate MLP and tied faces.

    Neither recorded actions nor expected leaf codes are inputs. Stops count
    as recovered codes only within numerical tolerance of a dictionary row;
    stopping on an off-codebook root is an explicit failure, not root snapping.
    """
    if type(order) is not int or order < 0:
        raise ValueError('unfolding requires the stored nonnegative order stamp')
    limit = min(int(limit), 128)
    weights = activation() if callable(activation) else activation
    if not torch.is_tensor(weights):
        return dict(codes=(), operations=(), spent=0, complete=False)
    weights = weights.reshape(-1)[:len(basis)]
    candidates = (weights > 1.).nonzero().flatten()
    if candidates.numel() > 32:
        candidates = candidates[torch.argsort(weights[candidates], descending=True, stable=True)[:32]]
    if candidates.numel() == 0:
        return dict(codes=(), operations=(), spent=0, complete=False)
    candidate_basis = basis[candidates.to(basis.device)]
    binary, unary = language._generate_binary_ops, language._generate_unary_ops
    inverses = language.reverse_inverses(binary)
    pending, codes, operations, spent, complete = [(idea.detach(), order)], [], [], 0, True
    while pending and spent < limit:
        if work is not None and not work.consume('unfold'):
            break
        value, current_order = pending.pop()
        top = value.reshape(1, -1)
        if not bool(torch.isfinite(top).all()):
            raise FloatingPointError('nonfinite idea during grammar unfolding')
        spent += 1
        action = int(language.generate_policy_logits(top).argmax(-1))
        index = torch.tensor([action], device=top.device)
        gate = torch.ones(1, dtype=torch.bool, device=top.device)
        if action < len(binary):
            left, right, unavailable = language.reverse_binary_step(
                top, index, gate, ops=binary, inverses=inverses, basis=candidate_basis,
                return_status=True)
            if bool(unavailable.any()):
                complete = False
                break
            operations.append(('binary', action))
            pending.extend(((right[0], current_order), (left[0], current_order)))
        elif action < len(binary) + len(unary):
            value, unavailable = language.reverse_unary_step(
                top, index - len(binary), gate, ops=unary, return_status=True)
            if bool(unavailable.any()):
                complete = False
                break
            operations.append(('unary', action - len(binary)))
            pending.append((value[0], current_order))
        else:
            # Terminal words can only name currently active symbols.
            distance = (candidate_basis.detach().to(top) - top).norm(dim=-1)
            if order_of is not None:
                eligible = distance.new_tensor([order_of(int(code)) <= current_order
                                                 for code in candidates], dtype=torch.bool)
                distance = distance.masked_fill(~eligible, torch.inf)
            value, row = distance.min(0)
            if float(value) > 1e-4 * max(1., float(top.norm())):
                complete = False
                break
            code = int(candidates[row])
            # Named abstractions are themselves valid retrieval terms. Their
            # lower-order witnesses are unfolded under the same work budget.
            codes.append(code)
            current_order = current_order if order_of is None else order_of(code)
            if current_order:
                children = () if sigma_inverse is None else sigma_inverse(code, current_order)
                if not children:
                    complete = False
                    break
                operations.append(('sigma', current_order, code))
                pending.extend((point, current_order - 1) for point in reversed(children))
            else:
                operations.append(('code', code))
    return {'codes': tuple(codes), 'operations': tuple(operations), 'spent': spent,
            'complete': complete and not pending}


def configure_model_index(model, store):
    """Bind the existing CS and generate owners before a closing or reader call."""
    if store is None or getattr(store, '_index_owner_bound', False):
        return
    from Queries import _existing_row, _basis
    registry = getattr(model, 'grammatical_thoughts', None) or getattr(getattr(model, 'symbolSpace', None), 'grammatical_thoughts', None)
    space = getattr(registry, 'space', None) or getattr(model, 'conceptualSpace', None)
    language = getattr(model, 'languageSpace', None)
    if space is None or getattr(space, 'similarity_codebook', None) is None:
        return
    def code_row(reference):
        try:
            return _existing_row(space, reference)
        except ValueError:
            return None
    def sigma_inverse(code, order):
        # Use the existing sigma inverse's learned case choice. The stamp
        # determines how many rungs must be decoded before a terminal read.
        from Spaces import _concept_alloc_of
        sigma = _concept_alloc_of(space).layer(0)
        basis = _basis(space)
        request = basis.new_zeros(sigma.nOutput, 1)
        if not 0 <= code < len(request):
            return ()
        request[code] = 1.
        attributed = sigma.attribute_presence(request, start=code, end=code + 1)[:, 0]
        width = sigma.nOutput
        signed = attributed[:width] - attributed[width + 1:2 * width + 1]
        chosen = (signed != 0).nonzero().flatten().tolist()
        if not chosen or any(space._row_order(row) != order - 1 for row in chosen):
            return ()
        return tuple(basis[row] * signed[row] for row in chosen)
    def unfold(value, limit, *, order=0, work=None):
        result = unfold_idea(language, _basis(space), value, limit, work=work,
                             activation=space.priming_weights, order=order,
                             order_of=space._row_order, sigma_inverse=sigma_inverse)
        return result['codes'], result['spent'], result['complete']
    store.configure_leaf_index(code_row=code_row, unfold=unfold if language is not None else None,
                               order_of=space._row_order)
    from ClauseRow import attach_clause_index
    attach_clause_index(model, store, space)
    import weakref
    book = space.similarity_codebook
    callbacks = getattr(book, '_row_remap_observers', None)
    if callbacks is None:
        callbacks = []
        object.__setattr__(book, '_row_remap_observers', callbacks)
    callbacks.append(weakref.WeakMethod(store.remap_leaf_codes))
    store._index_owner_bound = True
    if getattr(store, "_leaf_needs_owner_reindex", False):
        store.reindex_meanings()
