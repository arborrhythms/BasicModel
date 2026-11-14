"""Recall by an operator tree's extension, and two distinct truth readings."""
from dataclasses import replace
import torch
from Extension import ConceptSet, Context, Node, combine


def matches(store, forest, *, start=None):
    """Evaluate each operand over the whole row window, then its operators.

    A stored against reading opposes the positive question, and vice versa.
    No row is decoded or visited in the sparse tensor product.
    """
    from EpisodicMemory import matches as product
    from FieldLogic import reduce
    start = max(0,len(store)-store.ltm_window) if start is None else start
    concepts, values = store.episodic_concepts[start:len(store)],store.episodic_values[start:len(store)]
    indices = torch.arange(start,len(store),device=concepts.device)
    owner_ref = store.__dict__.get('_concept_owner')
    owner = None if owner_ref is None else owner_ref()
    def operand(field):
        if not len(field.ids) or field.missing:
            return ConceptSet(indices,values.new_zeros(len(indices),2),field.missing)
        orders = field.ids.new_tensor([0 if owner is None else -1 if owner._csw_row_of(int(cid)) is None
            else owner._row_order(owner._csw_row_of(int(cid))) for cid in field.ids])
        positive,present = product(concepts,values,field.ids,field.values,orders)
        negative,_ = product(concepts,-values,field.ids,field.values,orders)
        poles = reduce(torch.stack((positive,negative),-1),operation='intersection',dim=1,mask=present)
        return ConceptSet(indices,poles)
    def visit(node):
        if node.context.open:
            return ConceptSet(indices,values.new_zeros(len(indices),2),True)
        if node.operator is None:
            return operand(node.field)
        return node.operator.compose_extend(*(visit(child) for child in node.children),
            context=replace(node.context,rows=True))
    result = tuple(visit(root) for root in forest)
    return (combine(result,'union') if result else ConceptSet(indices,values.new_zeros(len(indices),2)))


def field_forest(field, stream=0):
    ids = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:,stream]
    values = (field.evidence[:,stream].amax(1) if field.evidence.shape[2]
              else field.evidence.new_zeros(len(ids),2))
    return (Node(ConceptSet(ids,values)),)


def forest(owner, meaning, *, work=None):
    """Unrecorded ideas use the ordinary generate walk, never a stored tree."""
    if meaning.extension:
        return meaning.extension
    model = owner._model
    if model is None or not callable(getattr(model,'generate',None)):
        return ()
    if work is not None and not work.remaining:
        work.require('generate')
    from Generative import walk_bank
    from Output import PreparedGeneration
    root = meaning.roles[None]
    bank = walk_bank(model,root,None)
    prepared = PreparedGeneration((),root,bank,None,torch.is_grad_enabled())
    tick = model.present()
    generated = model.generate(root,structure=None,occupied=meaning.role_mask[None],
        infix_rows=(True,),walk_options={},prepared=prepared,
        references=(tuple(ref if ref is not None and ref[0] == 'sym' else None for ref in meaning.role_refs),),
        when=torch.tensor([[[tick,tick+1]]],device=root.device),work=work)
    if bool(generated.truncated.any()):
        if work is not None and not work.remaining:
            from QueryWork import QueryWorkExhausted
            raise QueryWorkExhausted('shared query work_budget exhausted during generation')
        return ()
    return generated.extension[0]


def answer(owner, question, recalled, *, work=None):
    """Read the answer only at a matching operator path's open operand."""
    requested = forest(owner,question,work=work)
    available = forest(owner,recalled,work=work)
    found = []
    for root_index, root in enumerate(requested):
        if root_index >= len(available):
            continue
        for path in root.open_paths():
            value = available[root_index].operand(path)
            if value is not None and bool(value.values.any()):
                found.append(value)
    if not found:
        return None
    from FieldProjection import key
    from Extension import as_field
    from Meaning import ConceptualMeaning
    value = combine(tuple(found),'union')
    where = value.ids.new_zeros(1,1,2)
    point = key(owner,as_field((value,),where=where,when=where))[0,0]
    active = value.ids[value.values.amax(-1) > 0]
    reference = ('sym',int(active[0])) if len(active) == 1 else None
    return replace(ConceptualMeaning.from_description(point),role_refs=(reference,None,None),
                   extension=(Node(value,reference=reference),))


def existence(store, results):
    """Brightest witnessed row, plus the exact-content part usable as proof."""
    indices, scores = results.ids, results.values
    trust = store.trust[indices]
    lanes = torch.where(trust[:,None] < 0,scores.flip(-1),scores) * trust.abs()[:,None]
    exact = torch.where(scores == 1, torch.ones_like(scores),0.)
    exact = torch.where(trust[:,None] < 0,exact.flip(-1),exact) * trust.abs()[:,None]
    zero = scores.new_zeros(2)
    return (lanes.amax(0),exact.amax(0)) if len(indices) else (zero,zero)


def orient_relation(meaning, result):
    """Read a relation's proof on the requested positive or negative form.

    A retained relation names the positive claim. Its query's outer unary
    operators still act on that claim, just as they act on entry matches.
    """
    from ThoughtReferences import bindings
    invert, withheld, expressed = False, False, False
    if len(meaning.extension) == 1:
        node = meaning.extension[0]
        while node.operator is not None and len(node.children) == 1:
            operator = getattr(node.operator, 'gl', node.operator)
            effect = getattr(operator, 'polarity_effect', None)
            if effect == 'invert':
                invert, expressed = not invert, True
            elif effect == 'exclude':
                withheld, expressed = True, True
            node = node.children[0]
    if not expressed:
        invert = (not meaning.polarity) != bool(bindings(meaning).get('_operand_negated'))
    if withheld:
        return dict(result,support_true=0.,support_false=0.,path=(),against_path=(),
                    witnesses=(),frames=(),value=())
    if invert:
        return dict(result,support_true=result['support_false'],support_false=result['support_true'],
                    path=result.get('against_path',()),against_path=result.get('path',()))
    return result


def pervasion(owner, meaning, *, work, max_records, max_steps=8):
    """A higher-order relation covers the subject; members are never a census."""
    from Memory import SYMBOL
    from ConceptHierarchy import classes
    from Taxonomy import capture_taxonomy
    subject,prop = meaning.role_refs[0],meaning.role_refs[2]
    if subject is None or prop is None or subject[0] != 'sym' or prop[0] != 'sym':
        return dict(support_true=0.,support_false=0.,path=(),records_scanned=0,incomplete=())
    alloc = owner._concept_allocator
    kinds = classes(owner)
    targets = ((prop[1],) if prop[1] in kinds else tuple(cid for cid in kinds
        if alloc.relate_idx.get(('part-base',cid)) == (SYMBOL,prop[1])))
    if not targets:
        return dict(support_true=0.,support_false=0.,path=(),records_scanned=0,incomplete=())
    view = capture_taxonomy(owner,max_nodes=max_records,max_records=max_records,
                            focus=(subject,*(('sym',cid) for cid in targets)),work=work)
    results = [view.part_of(subject,('sym',cid),max_steps=max_steps,max_expansions=max_records,work=work)
               for cid in targets]
    best = max(results,key=lambda item:item['support_true'])
    against = max(results,key=lambda item:item['support_false'])
    return orient_relation(meaning,dict(best,support_false=against['support_false'],
                                       against_path=against.get('against_path',())))
