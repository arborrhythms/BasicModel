"""Read-only graph capture, including journals deeper than Python's stack."""
from arrival_capture import plain
from arrival_keep_probe import reading


def clause_graph(root):
    nodes, terms, seen, composed = [], [], {}, {}
    def number(item, records, indices, pending):
        if item is None:
            return None
        if id(item) not in indices:
            indices[id(item)] = len(records)
            records.append(None)
            pending.append(item)
        return indices[id(item)]
    pending_nodes, pending_terms = [], []
    root_id = number(root, nodes, seen, pending_nodes)
    while pending_nodes:
        node = pending_nodes.pop()
        request = node.reference_request
        nodes[seen[id(node)]] = dict(relation=node.relation, refs=node.refs,
            order=node.order, minted=node.minted, meaning=reading(node.meaning),
            subject_word_id=node.subject_word_id,
            request=None if request is None else {key: request.get(key)
                for key in ('mode', 'role', 'leaves', 'forward_identity')},
            children=[number(child, nodes, seen, pending_nodes) for child in node.children],
            companions=[number(child, nodes, seen, pending_nodes) for child in node.companions],
            composition=number(node.composition, terms, composed, pending_terms))
    while pending_terms:
        term = pending_terms.pop()
        terms[composed[id(term)]] = dict(reference=term.reference, word_id=term.word_id,
            operation=term.operation, formal_name=term.formal_name,
            head_role=term.head_role, order=term.order, leaves=term.leaves,
            children=[number(child, terms, composed, pending_terms) for child in term.children],
            binding=number(term.binding, terms, composed, pending_terms))
    return plain(dict(root=root_id, nodes=nodes, compositions=terms))
