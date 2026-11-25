"""Evaluate a completed grammatical formula on its referenced objects.

Extension.reading owns the traversal. This evaluator carries its numerical
operands alongside that walk; it neither chooses grammar nor grows a tree.
"""
from dataclasses import dataclass, replace, fields, is_dataclass

import torch
from Meaning import ConceptualMeaning, frozen_metadata


def detached(value, memo=None):
    """Detach a closing's existing records while preserving shared constituents."""
    memo = {} if memo is None else memo
    if type(value) is tuple and frozen_metadata(value):
        return value
    if id(value) in memo:
        return memo[id(value)]
    if torch.is_tensor(value):
        result = value.detach().clone()
    elif is_dataclass(value):
        result = replace(value, **{field.name: detached(getattr(value, field.name), memo)
                                   for field in fields(value)})
    elif isinstance(value, dict):
        result = {key: detached(item, memo) for key, item in value.items()}
    elif isinstance(value, (tuple, list)):
        result = type(value)(detached(item, memo) for item in value)
    else:
        return value
    memo[id(value)] = result
    return result


def from_stack(value, points, memo=None):
    """Keep native idea values and closing metadata, not numerical inner nodes."""
    from ClauseRow import Composition
    # Roles and their occupied lanes were formed before staging. Only the
    # composition carries points that can acquire a new STM address here.
    if isinstance(value, ConceptualMeaning):
        return value
    memo = {} if memo is None else memo
    if type(value) is tuple and frozen_metadata(value):
        return value
    if id(value) in memo:
        return memo[id(value)]
    if is_dataclass(value):
        # The operated extension already owns its live field and gradients.
        # Only composition points have to resolve through the staged STM.
        updates = {}
        for field in fields(value):
            if isinstance(value, Composition) and field.name == 'point':
                updates[field.name] = points.get(id(value))
            elif field.name != 'extension':
                updates[field.name] = from_stack(getattr(value, field.name), points, memo)
        result = replace(value, **updates)
    elif isinstance(value, dict):
        result = {key: from_stack(item, points, memo) for key, item in value.items()}
    elif isinstance(value, (tuple, list)):
        result = type(value)(from_stack(item, points, memo) for item in value)
    else:
        return value
    memo[id(value)] = result
    return result


@dataclass
class Result:
    """The semantic close, independent of the word formula's numerical journal."""
    source: object
    clause: object
    held_final: object
    concept_ids: torch.Tensor
    end_state: torch.Tensor
    equality_terms: tuple = ()
    root: object = None
    store: object = None

    def detached(self):
        memo = {}
        return replace(self, clause=detached(self.clause, memo),
            held_final=None if self.held_final is None else self.held_final.detached(),
            concept_ids=self.concept_ids.detach().clone(), end_state=self.end_state.detach().clone(),
            equality_terms=detached(self.equality_terms, memo))

    def admitted(self, word_reading):
        # Only admission's native lexical aliases change here. Numerical
        # comprehension has already run on the scored path and is not repeated.
        before, after = self.source, word_reading
        aliases = {}
        if before.reference_ids is not None and after.reference_ids is not None:
            rows = torch.stack((before.concept_ids, after.reference_ids, after.concept_ids), -1)
            aliases = {old: new for old, new, base in rows.tolist() if new != base}
        result = self.detached()
        if not aliases:
            return result
        from ClauseRow import Clause, Composition
        memo = {}
        def rewrite(value):
            if id(value) in memo:
                return memo[id(value)]
            if isinstance(value, Composition):
                changed = replace(value, reference=aliases.get(value.reference, value.reference),
                    children=tuple(rewrite(child) for child in value.children),
                    binding=None if value.binding is None else rewrite(value.binding))
            elif isinstance(value, Clause):
                changed = replace(value, refs=tuple(aliases.get(ref, ref) if type(ref) is int else ref
                    for ref in value.refs), children=tuple(rewrite(child) for child in value.children),
                    companions=tuple(rewrite(child) for child in value.companions),
                    composition=None if value.composition is None else rewrite(value.composition))
            else:
                return value
            memo[id(value)] = changed
            return changed
        result.clause = rewrite(result.clause)
        return result


@dataclass
class Value:
    point: torch.Tensor
    reference: int
    order: int
    cases: object = None
    where: object = None
    when: object = None


class Evaluation:
    def __init__(self, model, program, stream, *, trust=1., store=None):
        self.model, self.program, self.stream = model, program, stream
        self.trust = float(trust)
        self.owner, self.language = model._concept_owner(), model.languageSpace
        self.stack = []
        if store is None and program.tree is not None:
            store = program.tree.ideas
        self.leaves = (torch.empty_like(program.leaves) if store is None
                       else store.word_rows(program.leaves))
        self.ids = program.concept_ids.clone()
        self.native_ids = self.ids.clone()
        self.orders = (torch.full_like(self.ids, -1) if program.leaf_orders is None
                       else program.leaf_orders.clone())
        self._frames = (torch.zeros_like(program.operation_values) if store is None
                        else store.operation_rows(program.operation_values))
        self.held_writes = None if program.held_writes is None else program.held_writes.tolist()
        self.refs = (program.actions.new_full((len(program.actions), 2), -1)
                     if program.operation_refs is None else program.operation_refs.clone())
        self.case_values = None if program.case_values is None else torch.zeros_like(program.case_values)
        self.case_selected = None if program.case_selected is None else torch.zeros_like(program.case_selected)
        self.timeline = None if program.timeline_actions is None else program.timeline_actions.clone()
        self.choices = []
        self.result = None
        self.equality_terms = []
        from HeldConcepts import Evaluation as HeldEvaluation
        self.held = HeldEvaluation(program.held_initial, model)
        self.extensions, self.composition_sites = {}, {}
        self.identity_values = program.identity_values
        self.timeline_values = program.timeline_values
        self.end_state = None
        self.closing = None
        if program.tree is not None:
            from ClauseJournal import ComprehensionClose
            from Models import _boundary_registry
            self.closing = ComprehensionClose(self.language, program, self, _boundary_registry(model))

    def _cases(self, identity):
        bank = self.program.case_bank
        return None if bank is None else bank.view(self.ids.new_tensor([identity]))

    def native_leaf_reference(self, identity, original):
        if identity > 0 and self.owner._concept_row_of(identity) is None:
            return original
        return identity

    def leaf(self, word):
        program, owner = self.program, self.owner
        value = program.leaves[word]
        original = identity = int(program.concept_ids[word])
        lexical = -1 if program.word_ids is None else int(program.word_ids[word])
        if (lexical > 0 and program.interpret_forced and
                (program.staged_object_rows is None or int(program.staged_object_rows[word]) >= 0)):
            identity = owner.interpret.referent(lexical, batch=self.stream)
            row = owner._concept_row_of(identity)
            if identity == lexical and program.staged_object_ids is not None:
                pending = int(program.staged_object_ids[word])
                if pending > 0:
                    identity = pending
                    row = int(program.staged_object_rows[word])
            if row is None or row < 0:
                raise ValueError('comprehension referent has no staged conceptual row')
            atoms = owner.similarity_codebook.lookup_rows(row, stream=self.stream)
            atoms = owner.interpret.lexical_atoms(row, atoms, word_ids=lexical, stream=self.stream)
            evidence = (program.leaf_evidence[word] if program.leaf_evidence is not None else
                        torch.stack((program.activations[word].detach().to(value), value.new_zeros(()))))
            value = owner.interpret.forward(value, object_atoms=atoms,
                presence=program.activations[word], evidence=evidence)
            # A pending name has the word row's order, as reading() gives it;
            # its MTM handle is not an allocated concept whose order can be read.
            order = owner._row_order(row)
            self.orders[word] = order
        else:
            order = int(self.orders[word])
        self.choices.append((lexical, tuple(owner.interpret.denotations(lexical)) if lexical > 0 else (), identity))
        self.leaves[word] = value
        self.ids[word] = identity
        self.native_ids[word] = self.native_leaf_reference(identity, original)
        # Occurrence bands belong to the input location, not to the dictionary.
        point = value.clone()
        start = self.model.symbolSpace.ltm_store.when_start
        # A native word push has zero occurrence coordinates when this
        # modality supplied no bands. Carry those same coordinates through
        # comprehension, including publications of a newly opened identity.
        where = value.new_zeros(4) if program.symbol_where is None else program.symbol_where[word]
        when = value.new_zeros(4) if program.symbol_when is None else program.symbol_when[word]
        if start is not None:
            for band, offset in ((where, start-4), (when, start)):
                if offset >= 0:
                    point[offset:offset+4] = band
        self.stack.append(Value(point, identity, order, self._cases(identity),
            where, when))
        return identity

    def transpose(self):
        self.stack[-2:] = reversed(self.stack[-2:])

    def operation(self, at, kind, local, word, rule, operator):
        from ReferenceContext import denoted_order, OrderReference
        from Queries import StructuralGrammarContext, ConceptualSpaceCapability
        from ChunkJournal import action_context
        offset = max(0, word) if kind == 2 else 0
        if offset:
            self.stack.append(self.stack.pop(-1-offset))
        arity = 2 if kind == 1 and rule.method_name != 'bind' else 1
        children = self.stack[-arity:]; del self.stack[-arity:]
        operands, refs, orders, cases = [], [], [], []
        for role, child in enumerate(children):
            selected = (self.program.operation_resolved is not None
                        and bool(self.program.operation_resolved[at, role]))
            # A selected external operand is a recorded input, never a copied
            # operator result. Unsubstituted operands come from this evaluation.
            point = (self.program.operation_values[at, role] if selected else
                     child.point)
            if self.program.held_refresh is not None and not selected:
                refreshed = self.held.read(self.program.held_refresh[at,role])
                if refreshed is not None:
                    point = refreshed
            # Occurrence coordinates were read before the reference choice:
            # choose_operation reads relative time, then STMChunks.refresh can
            # take an off-stack value (whose occurrence band may be empty).
            # Read the recorded band directly. Computing a relative clock here
            # would only produce four coordinates discarded by this splice.
            start = self.model.symbolSpace.ltm_store.when_start
            if start is not None:
                low = max(0, start-4)
                point = torch.cat((point[:low], self.program.operation_values[at, role, low:start+4],
                                   point[start+4:]))
            identity = int(self.refs[at, role]) if selected else child.reference
            if selected:
                if self.program.operation_orders is None:
                    raise ValueError('a selected comprehension operand has no recorded order')
                # Pending MTM identities have no allocated concept row. Their
                # typed reference choice already recorded the operand's order.
                order = int(self.program.operation_orders[at, role])
            else:
                order = child.order
            operands.append(point[None]); refs.append(identity); orders.append(order)
            cases.append(self._cases(identity) if selected else child.cases)
            self.refs[at, role] = identity
        if rule.method_name == 'bind':
            bound = (None if self.program.held_bind is None else
                     self.held.read(self.program.held_bind[at],relative=True))
            operands.append(self.program.operation_values[at, 1:2] if bound is None else bound[None])
            refs.append(int(self.refs[at, 1]))
        head = 2 if getattr(rule, 'clause_form', None) == 'S' else getattr(rule, 'head_role', 0)
        case = cases[min(head-1, len(cases)-1)] if head else None
        selected_cases = None
        if getattr(rule, 'case_head_role', 0) and getattr(operator, 'meaning_layout', None) is None:
            from CaseSelection import select_cases
            if self.program.case_bank is None:
                raise ValueError('compound comprehension requires its recorded compose case bank')
            selected_cases = select_cases(self.program.case_bank, self.ids.new_tensor([refs[0]]),
                                         self.ids.new_tensor([refs[1]]))
        form = None
        if self.program.form_wholes is not None:
            number = self.program.form_whole_counts_raw
            if number is None:
                number = self.program.form_whole_counts
            form = {len(operands): (self.program.form_wholes[at:at+1], number[at:at+1],
                self.program.form_whole_presence[at:at+1], tuple(self.ids.new_tensor([v]) for v in orders)),
                'references': {len(operands): tuple(OrderReference(self.ids.new_tensor([v])) for v in refs)}}
        counts = None if self.program.operand_counts is None else self.program.operand_counts[at].unbind()
        context = StructuralGrammarContext(None, ConceptualSpaceCapability(operands[0].shape[-1]), None,
            'compose', selected_cases=selected_cases, leaf_counts=counts, form_context=form,
            action_context=action_context(self.program, at, operator), case_view=case)
        from EqualityLearning import observation
        equality = observation(rule, operands)
        if equality is not None:
            self.equality_terms.append(equality)
        # The selected forward in OperationSelectionLayer separates occurrence
        # coordinates from content before applying the cube kernel. The journal
        # keeps the unstripped operands; the comprehension must do both as well.
        numerical = (tuple(operands) if rule.method_name == 'bind' else
                     tuple(operator.content_operand(point) for point in operands))
        value = operator.compose_from_grammar_context(numerical, context=context)[0]
        # choose_operation restores the physical head's absolute time after
        # the numerical face. The next operation reads that time as age.
        physical = children[1] if len(children) == 2 and getattr(rule, 'head_role', 0) == 2 else children[0]
        start = self.model.symbolSpace.ltm_store.when_start
        if start is not None and physical.when is not None:
            value = torch.cat((value[:start], physical.when, value[start+4:]))
        if case is not None and getattr(rule, 'case_head_role', 0) and getattr(operator, 'meaning_layout', None) is not None:
            # The same selection used by CaseFlow.apply after a kept action.
            case = operator.select_view(operands[0], case)
        if self.case_values is not None and case is not None:
            layout = getattr(operator, 'meaning_layout', None)
            width = 0 if layout is None else layout[0]
            self.case_values = self.case_values.index_copy(0, self.ids.new_tensor([at]),
                torch.nn.functional.pad(case.fold(meaning_start=width)[0], (width, 0))[None])
            self.case_selected = self.case_selected.index_copy(0, self.ids.new_tensor([at]), case.selected[:1])
        order = denoted_order(rule, tuple(orders), tuple(OrderReference(v) for v in refs[:arity]), None)
        reference = refs[head-1] if head and head <= len(refs) else -1
        if rule.method_name == 'bind' and refs[-1] not in (-1, 0):
            reference = refs[-1]
        # Append to this candidate's reserved action slab. Consumers that
        # survive another write keep the three live operands for this action.
        frame = torch.stack((operands[0][0], operands[1][0] if len(operands) == 2 else value*0, value))
        self._frames[at].copy_(frame)
        self.current_frame = frame
        if self.held_writes is not None:
            for witness in self.held_writes[at]:
                self.held.publish(witness,frame[:2],value,self._frames,physical)
        if self.timeline is not None:
            self.timeline[at, 0] = order
        working = value
        band = getattr(getattr(self.language, 'chunk_journal', None), 'percept_band', None)
        if band is not None and value.shape[-1] == band[2] and physical.where is not None and physical.when is not None:
            working = torch.cat((value[:band[0]], physical.where, physical.when, value[band[1]:]))
        self.stack.append(Value(working, reference, order, case, physical.where, physical.when))
        if offset:
            self.stack.insert(len(self.stack)-1-offset, self.stack.pop())
        return reference

    def completed(self, at, extension=None):
        if extension is not None:
            self.extensions[at] = extension
        if self.closing is not None:
            self.closing.consume(at)

    def extension_at(self, *, action=None, word=None):
        if action is None:
            action = self.program.action_index.word_actions[word]
        node = self.extensions.get(action)
        return () if node is None else (node,)

    @property
    def frames(self):
        """An external snapshot may be differentiated after later appends."""
        return self._frames.clone()

    def closing_value(self, name):
        return getattr(self, dict(leaf_orders='orders', operation_values='_frames',
            operation_refs='refs', timeline_actions='timeline', case_values='case_values',
            case_selected='case_selected', identity_values='identity_values',
            timeline_values='timeline_values', end_state='end_state')[name])

    def composition(self, value, *, action=None, word=None):
        tree = self.program.tree
        address = tree.composition(action=action, word=word).address
        root_address = tree.roots[0]
        # The prior pinned meaning view read its root from the ended STM
        # slot, including the head's physical occurrence coordinates.
        if address == root_address and self.stack:
            value = replace(value, point=self.stack[0].point)
        value = replace(value, extension=self.extension_at(action=action, word=word))
        self.composition_sites[id(value)] = address
        return value

    def finish(self, *, roots=None):
        end = torch.zeros_like(self.program.end_state)
        for slot, node in enumerate(reversed(self.stack)):
            end[slot] = node.point
        self.end_state = end
        identities = self.program.identity_ids
        self.identity_values = (None if self.program.identity_values is None else
            self.held.identity_values(identities, 0, self.program.identity_values))
        self.timeline_values = (None if self.program.timeline_values is None else
            self.held.identity_values(identities, 1, self.program.timeline_values))
        clause = None if self.closing is None else self.closing.finish()
        self.result = Result(self.program, clause, self.held.concepts,
                             self.native_ids.clone(), end, tuple(self.equality_terms))
        if clause is not None:
            from PartClosing import PartClosing
            from Extension import with_extensions
            clause = with_extensions(clause, roots)
            store = self.program.tree.ideas
            source = self.model.symbolSpace.ltm_store
            store.to(device=end.device, dtype=end.dtype)
            store.prepare(source, self.program.tree, self.composition_sites)
            # The tree into the trial's STM before the closing stages into
            # it: each word a row naming its concept, each step its edge.
            # Prefer the word the writer can spell, then its resolved identity.
            # Both must name a seated concept at this point in the trial.
            words = self.program.word_ids
            names = torch.stack((self.ids if words is None else words, self.ids), -1)
            seated = (names > 0) & (self.owner._concept_allocator.layer(0).concept_rows(names) >= 0)
            concepts = torch.where(seated[:, 0], names[:, 0], torch.where(seated[:, 1], self.ids, -1))
            self.program.tree.write_edges(store, concepts,
                                          self.program.symbol_where, self.program.symbol_when, self.language)
            closing = PartClosing(self.owner, source, working=store)
            root, entries = closing.stage(clause, trust=self.trust)
            arrivals = self.held.retain_arrivals(clause,
                source.__dict__.get('_part_read_rules', ()), closing.staged_points)
            self.result.clause = from_stack(clause, {**arrivals.values, **closing.staged_points})
            self.result.root, self.result.store = root, store
        return self.result
