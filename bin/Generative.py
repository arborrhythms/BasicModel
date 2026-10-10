"""The shared, bounded inverse menu for stored ideas and sentence generation.

The word bank supplies terminals. Optional structural candidates supply
nonterminals; neither a compose journal nor a desired output enters a menu.
All binary hypotheses are checked through their actual forward kernel.
"""
from typing import NamedTuple
from functools import wraps
import torch


class TypeFamilies(NamedTuple):
    """A, B and native property wholes, in their existing conceptual chart."""
    codes: torch.Tensor
    valid: torch.Tensor
    families: torch.Tensor


def preserve_operator_activations(function):
    """Keep a host-side stored read from replacing live diagnostic state.

    Native numerical kernels publish detached activations during eager
    composition. Their values are useful to the owning forward, but inverse
    candidate evaluation must not publish them as another observed event.
    """
    @wraps(function)
    def read(language, *args, **kwargs):
        saved = {}
        for op in (*language._generate_binary_ops, *language._generate_unary_ops):
            for module in (op.modules() if isinstance(op, torch.nn.Module) else (op,)):
                if 'activation' in vars(module):
                    saved[id(module)] = (module, module.activation)
        try:
            return function(language, *args, **kwargs)
        finally:
            for module, activation in saved.values():
                module.activation = activation
    return read


@torch.no_grad()
def determiner_expansions(language, parent, order, basis, orders, *, binding=None):
    """Realize a lower-order referent through a higher-order description.

    A determiner need not change the point: increasing the description's
    order is progress in this inverse. Native order and binding facts supply
    that context; neither a recorded operation nor its probability is read.
    Marker words are ranked by the existing compose grammar on each pair.
    A bare point, without an order mismatch, licenses no such expansion.
    """
    rules = getattr(language, '_generate_binary_rules', ())
    compose = getattr(language, '_compose_binary_rules', ())
    layer = getattr(getattr(language, 'language_layer', None), 'operation_layer', None)
    if layer is None or not len(basis):
        return {}
    distance = (basis - parent).norm(dim=-1)
    descriptions = (distance <= 1e-4 * max(1., float(parent.norm()))) & (orders == order + 1)
    if not bool(descriptions.any()):
        return {}
    result = {}
    for index, rule in enumerate(rules):
        mode = getattr(rule, 'determiner_mode', None)
        if mode not in ('mint', 'bind') or (binding is not None and mode != binding):
            continue
        key = language._generate_rule_key(rule, 2)
        matches = [i for i, candidate in enumerate(compose)
                   if language._generate_rule_key(candidate, 2) == key]
        if len(matches) != 1:
            continue
        operation = matches[0]
        op = language._generate_binary_ops[index]
        description = basis[descriptions.nonzero()[0, 0]]
        # This is a bounded reverse application of the ordinary learned
        # word/operator association, with no names or vocabulary table.
        pairs = torch.stack((basis, description.expand_as(basis)), 1)
        folded = op.compose(pairs[:, 0], pairs[:, 1])
        _, scores = layer.chooser.score_binary(
            pairs[..., :layer.d_model], folded[:, None, None, :layer.d_model],
            layer.stop_anchor, layer.reduce_anchor[operation:operation+1],
            op_indices=torch.tensor([operation], device=basis.device))
        scores = scores.reshape(-1)
        finite = (torch.isfinite(scores) & torch.isfinite(basis).all(-1) & basis.ne(0).any(-1)
                  & (orders >= 0)
                  & ((folded-parent).norm(dim=-1) <= 1e-4 * max(1., float(parent.norm()))))
        if bool(finite.any()):
            marker = int(scores.masked_fill(~finite, -torch.inf).argmax())
            result[index] = (basis[marker], description, int(orders[marker]), order + 1)
    return result


class LiveConstituents:
    """An invocation-only, STM-sized inverse snapshot of composed wholes.

    Values are detached forward activations. No source positions, actions,
    child pointers or original words survive in this unordered candidate set.
    Its lifetime ends with the sentence trial; it never enters LTM.
    """
    def __init__(self, words, word_valid, capacity):
        self.words, self.word_valid = words.detach(), word_valid
        self.codes = words.new_zeros(words.shape[0], int(capacity), words.shape[-1])
        self.valid = torch.zeros(self.codes.shape[:2], device=words.device, dtype=torch.bool)

    @torch.no_grad()
    def observe(self, state, rows, support):
        values, depth = state[:2]
        for b in rows.nonzero().flatten().tolist():
            for slot, value in enumerate(values[b, :int(depth[b])].detach()):
                if support is not None and int(support[b, slot].count_nonzero()) < 2:
                    continue
                if (not bool(value.any()) or not bool(torch.isfinite(value).all())
                        or bool((self.word_valid[b] & self.words[b].eq(value).all(-1)).any())
                        or bool((self.valid[b] & self.codes[b].eq(value).all(-1)).any())):
                    continue
                # Recency determines eviction only. Pair search is invariant
                # to this order except genuinely indistinguishable inverse ties.
                self.codes[b] = torch.cat((self.codes[b, 1:], value[None]), 0)
                self.valid[b] = torch.cat((self.valid[b, 1:], self.valid.new_ones(1)), 0)


@torch.no_grad()
def primed_type_families(space, heat, *, limit):
    """Snapshot existing types by family, without minting or reading episodes.

    Dictionary membership is learned. There is no word/POS table: A supplies
    objects, B supplies changes in A's coordinates, and other higher-order
    concepts supply property wholes. Cold and absent columns stay unavailable.
    Each family has the existing reconstruction candidate bound.
    """
    if heat.ndim == 1:
        heat = heat[None]
    B, _ = heat.shape
    book = space.similarity_codebook
    width = book.W.shape[-1]
    rows = heat.new_full((B, 3 * int(limit)), -1, dtype=torch.long)
    components = getattr(space, 'components', None)
    nouns = set(() if components is None else components.nouns.ids)
    verbs = set(() if components is None else components.verbs.ids)
    for b in range(B):
        grouped = [[], [], []]
        for row in (heat[b] > 1).nonzero().flatten().tolist():
            identity = space.concept_id_at_row(row)
            if identity is None:
                continue
            family = 0 if identity in nouns else 1 if identity in verbs else 2
            if family == 2 and space._row_order(row) <= 0:
                continue
            grouped[family].append(row)
        for family, candidates in enumerate(grouped):
            candidates.sort(key=lambda row: (-float(heat[b, row]), row))
            candidates = candidates[:int(limit)]
            if candidates:
                rows[b, family * limit:family * limit + len(candidates)] = rows.new_tensor(candidates)
    valid = rows >= 0
    if bool(valid.any()):
        codes = book.lookup_rows(rows.clamp_min(0)).detach()
        interpretation = getattr(space, 'interpret', None)
        if interpretation is not None:
            codes = interpretation.binding_atoms(codes)
        codes = torch.where(valid[..., None], codes, 0.)
    else:
        codes = heat.new_zeros(B, 3 * int(limit), width)
    families = torch.arange(3, device=heat.device).repeat_interleave(int(limit))
    return TypeFamilies(codes, valid, families)


def reconstruction_coverage(emitted, expected):
    """Post-read support audit; counts never enter the inverse's decisions.

    A projection can perfectly preserve a head while losing its modifier.
    Numerical recomposition alone cannot certify that reading as complete.
    """
    missing, excess = (expected - emitted).clamp_min(0), (emitted - expected).clamp_min(0)
    return missing, excess, (missing + excess) == 0


def inverse_menu(language, parent, live, *, basis, basis_valid, priming=None,
                 limit=16, case_bank=None, constituents=None,
                 constituent_valid=None, constituent_families=None, require_symbols=True,
                 terminal_valid=None):
    from Language import LanguageSpace
    words, word_valid = basis, basis_valid
    terminals = word_valid if terminal_valid is None else word_valid & terminal_valid
    if constituents is not None:
        if basis is None or basis_valid is None or constituent_valid is None:
            raise ValueError('structural decoding requires explicit word and constituent masks')
        basis = torch.cat((basis, constituents.detach()), 1)
        basis_valid = torch.cat((basis_valid, constituent_valid), 1)
        priming = torch.cat((torch.ones_like(word_valid, dtype=parent.dtype)
                             if priming is None else priming,
                             constituent_valid.to(parent)), 1)
    # Families describe candidate types. Every type is admitted on either
    # side; recomposition and the learned generate policy decide orientation.
    binary, unary = language._generate_binary_ops, language._generate_unary_ops
    lefts, rights, missing = [], [], []
    for i in range(len(binary)):
        if basis is None and require_symbols:
            left, right, unavailable = parent, parent, live
        else:
            left, right, unavailable = language.reverse_binary_step(
                parent, torch.full_like(live, i, dtype=torch.long), live,
                ops=binary, basis=basis if require_symbols else None,
                basis_valid=basis_valid if require_symbols else None,
                basis_priming=priming, candidate_limit=limit,
                free=require_symbols, return_status=True, case_bank=case_bank)
        lefts.append(left); rights.append(right); missing.append(unavailable)
    for i in range(len(unary)):
        value, unavailable = language.generate_unary_step(
            parent, torch.full_like(live, i, dtype=torch.long), live, return_status=True)
        unavailable = unavailable | ~value.detach().ne(parent.detach()).any(-1)
        lefts.append(value); rights.append(parent); missing.append(unavailable)
    lefts.append(parent); rights.append(parent); missing.append(torch.zeros_like(live))
    available = ~torch.stack(missing, 1)
    # Priming can name an unspelled structural whole. Keep it in the
    # forward search while restricting only STOP to lexical realizations.
    options = ({} if constituents is None and terminal_valid is None else
               dict(terminal_basis=words, terminal_valid=terminals))
    legal = (LanguageSpace.decoder_eligibility(parent, lefts, rights, available,
        binary, basis, basis_valid, case_bank=case_bank, **options)
        if require_symbols else available)
    return torch.stack(lefts, 1), torch.stack(rights, 1), legal
