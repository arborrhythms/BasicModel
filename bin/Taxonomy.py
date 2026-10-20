"""Bounded read views of ConceptualSpace's existing conceptual taxonomy.

No memory is admitted here. A view copies typed reference records at one
boundary and provides hard structural evidence, never geometric world truth.
"""
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType


def concept_reference(value):
    """Validate an existing typed concept handle, never interpret a tensor row."""
    if (not isinstance(value, tuple) or len(value) != 2 or value[0] != "sym"
            or type(value[1]) is not int or value[1] <= 0):
        raise TypeError("taxonomy operands require a ('sym', positive concept id) reference")
    return value


def _limit(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class TaxonomyEdge:
    part: tuple
    whole: tuple
    owner: tuple
    role: str
    evidence: tuple = (1., 0.)


@dataclass(frozen=True)
class ConceptualTaxonomyView:
    """An invocation-owned derived view, not another authoritative taxonomy.

    A recorded sym-part/sym-whole link states structural inclusion in this
    domain. Sparse transform weights are not interpreted as fact confidence.
    Missing paths remain unknown. A supported path is not a world assertion.
    """
    concepts: frozenset
    edges: tuple
    nodes_scanned: int = 0
    records_scanned: int = 0
    incomplete: tuple = ()
    _outgoing: object = field(init=False, repr=False)
    _incoming: object = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "concepts", frozenset(self.concepts))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "incomplete", tuple(self.incomplete))
        outgoing, incoming = {}, {}
        for edge in self.edges:
            outgoing.setdefault(edge.part, []).append(edge)
            incoming.setdefault(edge.whole, []).append(edge)
        object.__setattr__(self, "_outgoing", MappingProxyType(
            {ref: tuple(edges) for ref, edges in outgoing.items()}))
        object.__setattr__(self, "_incoming", MappingProxyType(
            {ref: tuple(edges) for ref, edges in incoming.items()}))

    def neighbors(self, reference, *, direction="up"):
        ref = concept_reference(reference)
        if direction not in ("up", "down"):
            raise ValueError("taxonomy direction must be up or down")
        source = self._outgoing if direction == "up" else self._incoming
        return source.get(ref, ())

    def part_of(self, part, whole, *, max_steps=8, max_expansions=1024,
                work=None):
        """Return a bounded inclusion proof and its actual native record sources.

        Reification records A -> relation-concept -> B count as two links.
        Cycles are visited once. Re-reading a link cannot increase support.
        """
        part, whole = concept_reference(part), concept_reference(whole)
        max_steps = _limit(max_steps, "max_steps")
        max_expansions = _limit(max_expansions, "max_expansions")
        incomplete = list(self.incomplete)
        result = {"domain": "conceptual-taxonomy", "support_true": 0.0,
                  "support_false": 0.0, "path": (), "incomplete": (),
                  "nodes_scanned": self.nodes_scanned,
                  "records_scanned": self.records_scanned, "edges_expanded": 0}
        if part not in self.concepts or whole not in self.concepts:
            incomplete.append("unavailable_reference")
        elif part == whole:
            result["support_true"] = 1.0  # reflexive inclusion of a known concept
        else:
            queue = deque([(part, (), 1.)])
            visited = {part: 1.}
            stopped = False
            while queue and not stopped:
                current, path, strength = queue.popleft()
                outgoing = self._outgoing.get(current, ())
                if len(path) >= max_steps:
                    if outgoing:
                        incomplete.append("traversal_limit")
                    continue
                for index in range(len(outgoing)):
                    if result["edges_expanded"] >= max_expansions:
                        incomplete.append("traversal_limit")
                        stopped = True
                        break
                    if work is not None and not work.consume("expansion"):
                        incomplete.append("work_budget")
                        stopped = True
                        break
                    edge = outgoing[index]
                    result["edges_expanded"] += 1
                    proof = path + (edge,)
                    support = min(strength, edge.evidence[0])
                    if edge.whole == whole:
                        if support > result["support_true"]:
                            result["support_true"], result["path"] = support, proof
                        # A negative direct edge is counterevidence. It is
                        # never propagated through positive containment.
                        if current == part:
                            result["support_false"] = max(result["support_false"], edge.evidence[1])
                    if support > visited.get(edge.whole, 0.):
                        visited[edge.whole] = support
                        queue.append((edge.whole, proof, support))
        result["incomplete"] = tuple(dict.fromkeys(incomplete))
        return result


def capture_taxonomy(conceptual_space, *, max_nodes=256, max_records=1024,
                     focus=(), work=None):
    """Copy bounded conceptual reference records without creating any state.

    Endpoint definitions and reified relations share the existing allocator's
    record owner. Raw percept/whole codes, LTM relation rows and numeric
    codebook geometry are outside this reader. Optional focus handles are
    visited first; incidental node scans remain bounded independently of STM.
    """
    max_nodes = _limit(max_nodes, "max_nodes")
    max_records = _limit(max_records, "max_records")
    if not isinstance(focus, (tuple, list)):
        raise TypeError("taxonomy focus must be a finite reference sequence")
    truncated_focus = len(focus) > max_nodes
    focus = tuple(concept_reference(ref)[1] for ref in focus[:max_nodes])
    allocator = getattr(conceptual_space, "_concept_allocator", None)
    layer = None if allocator is None else getattr(allocator, "_layers", {}).get(0)
    if allocator is None or layer is None:
        return ConceptualTaxonomyView(frozenset(), (), incomplete=("unavailable_taxonomy",))
    placement, retired = allocator.placement, allocator.retired
    concepts, edges, visited = set(), [], set()
    nodes = records = 0
    seen_edges = set()
    incomplete = ["capture_limit"] if truncated_focus else []

    def order():
        yield from focus
        yield from placement  # no full-inventory copy/sort before the work bound

    def available(cid):
        return cid in placement and cid not in retired

    for cid in order():
        if cid in visited:
            continue
        if nodes >= max_nodes:
            incomplete.append("capture_limit")
            break
        if work is not None and not work.consume("node"):
            incomplete.append("work_budget")
            break
        visited.add(cid)
        nodes += 1
        if not available(cid):
            incomplete.append("unavailable_reference")
            continue
        owner = ("sym", int(cid))
        concepts.add(owner)
        part_store = allocator.parts
        incident = [part_store.find(part, cid) for part in part_store.parts(cid)]
        incident.extend(part_store.find(cid, whole) for whole in part_store.wholes(cid))
        for symbol in incident:
            if symbol in seen_edges:
                continue
            if records >= max_records:
                incomplete.append('capture_limit')
                break
            if work is not None and not work.consume('record'):
                incomplete.append('work_budget')
                break
            seen_edges.add(symbol)
            records += 1
            record = part_store.row(symbol)
            a, b = record['refs'].tolist()
            if not available(a) or not available(b):
                incomplete.append('unavailable_reference')
                continue
            part, whole = ('sym', a), ('sym', b)
            concepts.update((part, whole))
            evidence = (float(record['truth_plus']), float(record['truth_minus']))
            edges.append(TaxonomyEdge(part, whole, ('sym', symbol), 'part', evidence))
        if 'work_budget' in incomplete or 'capture_limit' in incomplete:
            break
    return ConceptualTaxonomyView(frozenset(concepts), tuple(edges), nodes, records,
                                  tuple(dict.fromkeys(incomplete)))
