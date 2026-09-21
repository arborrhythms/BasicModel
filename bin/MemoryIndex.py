"""Tensor-owned leaf-code columns and bounded cue retrieval for ternary LTM.

The posting lists are a derived index, never another semantic store. The row
vectors remain on TernaryTruthStore. No nearest-root shortcut supplies terms:
missing derivations must be unfolded by the configured grammar owner.
"""
from __future__ import annotations

import heapq
import itertools
import math

import torch
from torch.nn import functional as F


class LeafCodeIndex:
    """Mixin: the existing seal writer owns these columns and their lifetime."""

    def _init_leaf_index(self):
        self.register_buffer('leaf_codes', torch.empty(0, dtype=torch.long))
        self._leaf_used = 0
        self.register_buffer('leaf_offsets', torch.zeros(self.capacity, 3, 2, dtype=torch.long))
        self.register_buffer('leaf_complete', torch.zeros(self.capacity, 3, dtype=torch.bool))
        self.register_buffer('index_stream', torch.full((self.capacity,), -1, dtype=torch.long))
        self._leaf_postings = {}
        self._index_occurrences = {}
        self._index_code_row = None
        self._index_unfold = None
        self._leaf_needs_owner_reindex = False

    def configure_leaf_index(self, *, code_row=None, unfold=None):
        """Bind the codebook address adapter and bounded grammar unfold owner."""
        if any(value is not None and not callable(value) for value in (code_row, unfold)):
            raise TypeError('leaf index adapters must be callable')
        self._index_code_row, self._index_unfold = code_row, unfold

    @staticmethod
    def _checked_leaf_terms(terms):
        terms = tuple(tuple(values) for values in terms)
        if len(terms) != 3 or any(type(code) is not int or code < 0
                                  for values in terms for code in values):
            raise ValueError('leaf codes require three sequences of nonnegative code addresses')
        return terms

    def _meaning_leaf_terms(self, meaning, *, max_nodes=1024, work=None):
        """Resolve recorded leaves, then unfold only roles lacking a derivation."""
        remaining = [int(max_nodes)]
        active = set()

        def visit(value):
            if id(value) in active or remaining[0] <= 0:
                return ((), (), ()), (False, False, False)
            remaining[0] -= 1
            active.add(id(value))
            terms, complete = [], []
            for role, reference in enumerate(value.role_refs):
                codes, known = (), not bool(value.role_mask[role])
                if reference is not None and reference[0] == 'sym':
                    code = (self._index_code_row(reference) if self._index_code_row
                            else None)
                    if code is not None and int(code) >= 0:
                        codes, known = (int(code),), True
                elif reference is not None and reference[0] == 'constituent':
                    child, status = visit(value.constituents[reference[1]])
                    codes, known = tuple(itertools.chain.from_iterable(child)), all(status)
                elif reference is not None and reference[0] == 'ltm':
                    index = self._index_occurrences.get(reference)
                    if index is not None:
                        codes = tuple(itertools.chain.from_iterable(
                            self.leaf_terms(index, r) for r in range(3)))
                        known = bool(self.leaf_complete[index].all())
                if not known and self._index_unfold is not None and remaining[0] > 0:
                    # The callback receives only this detached idea, never the
                    # recorded answer leaves or a source reconstruction trace.
                    allowance = min(remaining[0], work.remaining) if work is not None else remaining[0]
                    recovered = (self._index_unfold(value.roles[role].detach(), allowance, work=work)
                                 if work is not None else self._index_unfold(value.roles[role].detach(), allowance))
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

        return visit(meaning)

    def leaf_terms(self, row, role):
        row, role = int(row), int(role)
        if not 0 <= row < len(self) or role not in (0, 1, 2):
            raise IndexError('leaf index row/role is unavailable')
        start, count = self.leaf_offsets[row, role].tolist()
        return tuple(self.leaf_codes[start:start + count].tolist())

    def rows_for_code(self, code, *, role=None):
        """Exact audit view; bounded readers use the posting iterators below."""
        if type(code) is not int or code < 0 or role not in (None, 0, 1, 2):
            raise ValueError('invalid leaf-code cue')
        if role is not None:
            return tuple(self._leaf_postings.get((code, role), ()))
        return tuple(sorted(set(itertools.chain.from_iterable(
            self._leaf_postings.get((code, r), ()) for r in range(3)))))

    @torch.no_grad()
    def _reserve_leaf_codes(self, size):
        """Geometric capacity makes successive writes amortized linear."""
        if size <= self.leaf_codes.numel():
            return
        capacity = max(64, 2 * self.leaf_codes.numel())
        while capacity < size:
            capacity *= 2
        column = self.leaf_codes.new_empty(capacity)
        column[:self._leaf_used].copy_(self.leaf_codes[:self._leaf_used])
        self.leaf_codes = column

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # Capacity is an allocation detail, not persisted index content. The
        # used prefix preserves the existing checkpoint column format. Clone
        # it: torch.save retains a view's entire backing allocation otherwise.
        value = self.leaf_codes[:self._leaf_used].clone()
        destination[prefix + 'leaf_codes'] = value if keep_vars else value.detach()

    @torch.no_grad()
    def _append_leaf_terms(self, row, terms, complete, stream=-1):
        terms = self._checked_leaf_terms(terms)
        if self._index_code_row is None and not all(complete):
            self._leaf_needs_owner_reindex = True
        flat = tuple(itertools.chain.from_iterable(terms))
        offset = self._leaf_used
        if flat:
            self._reserve_leaf_codes(offset + len(flat))
            self.leaf_codes[offset:offset + len(flat)] = self.leaf_codes.new_tensor(flat)
            self._leaf_used += len(flat)
        for role, codes in enumerate(terms):
            self.leaf_offsets[row, role] = self.leaf_offsets.new_tensor((offset, len(codes)))
            self.leaf_complete[row, role] = complete[role]
            offset += len(codes)
            for code in set(codes):
                self._leaf_postings.setdefault((code, role), []).append(row)
        self.index_stream[row] = int(stream)
        self._index_occurrences[self.occurrence_of(row)] = row

    def rebuild_leaf_postings(self):
        self._leaf_postings, self._index_occurrences = {}, {}
        for row in range(len(self)):
            self._index_occurrences[self.occurrence_of(row)] = row
            for role in range(3):
                start, count = self.leaf_offsets[row, role].tolist()
                if start < 0 or count < 0 or start + count > self._leaf_used:
                    raise ValueError('invalid leaf index offsets in checkpoint')
                codes = self.leaf_terms(row, role)
                if any(code < 0 for code in codes):
                    raise ValueError('negative leaf code in checkpoint')
                for code in set(codes):
                    self._leaf_postings.setdefault((code, role), []).append(row)

    @torch.no_grad()
    def reindex_meanings(self):
        """Rebuild a legacy index after its real codebook owner is available."""
        streams = self.index_stream.clone()
        previous = [tuple(self.leaf_terms(row, role) for role in range(3))
                    for row in range(len(self))]
        complete_before = self.leaf_complete.clone()
        self._leaf_used = 0
        self.leaf_offsets.zero_()
        self.leaf_complete.zero_()
        self._leaf_postings, self._index_occurrences = {}, {}
        for row in range(len(self)):
            meaning = self.meaning_of(row)
            if meaning is None:
                terms, complete = ((), (), ()), (False, False, False)
            else:
                terms, complete = self._meaning_leaf_terms(meaning)
            # Recorded derivation columns already hold real codebook rows.
            # Binding a late owner fills gaps; it must not replace that forest.
            terms = tuple(previous[row][role] if bool(complete_before[row, role])
                          else terms[role] for role in range(3))
            complete = tuple(bool(complete_before[row, role]) or complete[role]
                             for role in range(3))
            self._append_leaf_terms(row, terms, complete, int(streams[row]))
        self._leaf_needs_owner_reindex = False

    @torch.no_grad()
    def compact_leaf_rows(self, keep):
        """Compact columns with the same row permutation used by the store."""
        terms = [tuple(self.leaf_terms(int(row), role) for role in range(3)) for row in keep]
        statuses = self.leaf_complete[keep].clone()
        streams = self.index_stream[keep].clone()
        self._leaf_used = 0
        self.leaf_offsets.zero_()
        self.leaf_complete.zero_()
        self.index_stream.fill_(-1)
        # The caller publishes count/occurrences then rebuilds the postings.
        offset, flat = 0, []
        for row, roles in enumerate(terms):
            for role, codes in enumerate(roles):
                self.leaf_offsets[row, role] = self.leaf_offsets.new_tensor((offset, len(codes)))
                flat.extend(codes)
                offset += len(codes)
        self._reserve_leaf_codes(len(flat))
        self.leaf_codes[:len(flat)] = self.leaf_codes.new_tensor(flat)
        self._leaf_used = len(flat)
        self.leaf_complete[:len(keep)] = statuses
        self.index_stream[:len(keep)] = streams

    @torch.no_grad()
    def remap_leaf_codes(self, mapping):
        """Apply a codebook compaction map; an omitted code loses indexability."""
        mapping = dict(mapping)
        terms = []
        for row in range(len(self)):
            roles = []
            for role in range(3):
                old = self.leaf_terms(row, role)
                kept = tuple(int(mapping[code]) for code in old
                             if code in mapping and int(mapping[code]) >= 0)
                if len(kept) != len(old):
                    self.leaf_complete[row, role] = False
                roles.append(kept)
            terms.append(tuple(roles))
        complete, streams = self.leaf_complete.clone(), self.index_stream.clone()
        self._leaf_used = 0
        self._leaf_postings, self._index_occurrences = {}, {}
        for row, roles in enumerate(terms):
            self._append_leaf_terms(row, roles, complete[row], int(streams[row]))

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
        terms, _ = self._meaning_leaf_terms(cue, work=work)
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
def unfold_idea(language, basis, idea, limit, *, work=None):
    """Unfold a detached idea with the current generate MLP and tied faces.

    Neither recorded actions nor expected leaf codes are inputs. Stops count
    as recovered codes only within numerical tolerance of a dictionary row;
    stopping on an off-codebook root is an explicit failure, not root snapping.
    """
    limit = min(int(limit), 128)
    binary, unary = language._generate_binary_ops, language._generate_unary_ops
    inverses = language.reverse_inverses(binary)
    pending, codes, operations, spent, complete = [idea.detach()], [], [], 0, True
    while pending and spent < limit:
        if work is not None and not work.consume('unfold'):
            break
        top = pending.pop().reshape(1, -1)
        if not bool(torch.isfinite(top).all()):
            raise FloatingPointError('nonfinite idea during grammar unfolding')
        spent += 1
        action = int(language.generate_policy_logits(top).argmax(-1))
        index = torch.tensor([action], device=top.device)
        gate = torch.ones(1, dtype=torch.bool, device=top.device)
        if action < len(binary):
            left, right, unavailable = language.reverse_binary_step(
                top, index, gate, ops=binary, inverses=inverses, basis=basis,
                return_status=True)
            if bool(unavailable.any()):
                complete = False
                break
            operations.append(('binary', action))
            pending.extend((right[0], left[0]))
        elif action < len(binary) + len(unary):
            value, unavailable = language.reverse_unary_step(
                top, index - len(binary), gate, ops=unary, return_status=True)
            if bool(unavailable.any()):
                complete = False
                break
            operations.append(('unary', action - len(binary)))
            pending.append(value[0])
        else:
            # Compare terminal emissions only. Tile to bound working memory.
            best = (float('inf'), -1)
            for start in range(0, len(basis), 256):
                distance = (basis[start:start + 256].detach().to(top) - top).norm(dim=-1)
                value, row = distance.min(0)
                if float(value) < best[0]:
                    best = float(value), start + int(row)
            if best[0] > 1e-4 * max(1., float(top.norm())):
                complete = False
                break
            codes.append(best[1])
            operations.append(('code', best[1]))
    return {'codes': tuple(codes), 'operations': tuple(operations), 'spent': spent,
            'complete': complete and not pending}


def recorded_leaf_terms(program, meaning, depth):
    """Leaf rows of each retained role's actual forward derivation.

    Canonical relation roles use their native references/constituents. The
    fallback STM layout uses its completed forest, in older-to-newer order.
    """
    if program is None:
        return None
    if meaning is not None:
        from Meaning import ConceptualMeaning
        actual = ConceptualMeaning.from_payload(program.end_state, depth=depth, layout='stm')
        if not torch.equal(actual.roles, meaning.roles) or not torch.equal(actual.role_mask, meaning.role_mask):
            return None
    stack = []
    for kind, local, word, *_ in program.actions.detach().cpu().tolist():
        if kind < 0:
            break
        if kind == 0:
            code = int(program.rows[int(word)])
            if code < 0:
                return None
            stack.append((code,))
        elif kind == 1 and len(stack) >= 2:
            right, left = stack.pop(), stack.pop()
            stack.append(left + right)
        elif kind == 2 and stack:
            pass
        else:
            return None
    if len(stack) != int(depth) or not 1 <= len(stack) <= 3:
        return None
    # ConceptualMeaning.from_payload(layout='stm') maps newest-first slots
    # into [NP1, VP, NP2]. The program forest is oldest-first.
    if len(stack) == 3:
        return stack[1], stack[0], stack[2]
    return tuple(stack) + ((),) * (3 - len(stack))


def configure_model_index(model, store):
    """Bind the existing CS and generate owners before a seal or reader call."""
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
    def unfold(value, limit, *, work=None):
        result = unfold_idea(language, _basis(space), value, limit, work=work)
        return result['codes'], result['spent'], result['complete']
    store.configure_leaf_index(code_row=code_row, unfold=unfold if language is not None else None)
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
