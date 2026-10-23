"""FORCED decomposition grammar fixture, section 14.11.

The fixture selects legal compose choices through the actual chooser seam.
It does not replace a meaning, a driver, a result, a cost, a keep decision or
the live explore suffix. It is never imported by the measurement helpers.
"""
from contextlib import ExitStack
from unittest.mock import patch
import torch


class ForcedDecompositionGrammar:
    def __init__(self, model, *, open_names=(), named_bindings=None, generic_subjects=(), question_budget=None,
                 preferred_references=None, reference_orders=()):
        self.model, self.open_names = model, set(open_names)
        self.named_bindings = dict(named_bindings or {})
        self.generic_subjects = set(generic_subjects)
        self.question_budget = question_budget
        self.preferred_references = preferred_references
        self.reference_orders = set(reference_orders)
        self.word, self.current, self.data, self.texts = 0, None, None, ()
        self.records = []

    def __enter__(self):
        from SentenceFork import SentenceFork
        from Models import BasicModel
        self.stack = ExitStack()
        language = self.model.languageSpace
        layer = language.language_layer.operation_layer
        begin, batch = SentenceFork.begin_word, BasicModel.runBatch
        choose, forward, select = language.choose_operation, layer.forward, layer.select_logits
        resolver = self.model._concept_owner().components.reference_close
        select_reference = resolver.select
        owner = self.model._concept_owner()
        from Spaces import _concept_alloc_of
        allocation = _concept_alloc_of(owner)
        lookup = owner.interpret.lookup_word
        def lookup_word(*args, **kwargs):
            word = lookup(*args, **kwargs)
            form = kwargs.get('form')
            form = form.decode() if isinstance(form, bytes) else form
            if form in self.reference_orders:
                allocation.reference_orders[word] = 1
            return word
        for form, words in allocation.word_forms.items():
            form = form.decode() if isinstance(form, bytes) else form
            if form in self.reference_orders:
                for word in words:
                    allocation.reference_orders[word] = 1
        def reference_choice(choices, logits, *, forward_identity=None, individuals=None):
            # The same declared open-name choice applies when the closing
            # owner calls bind. In particular, the question's "what" must
            # remain anonymous until the checked thought episode fills it.
            owner = self.model._concept_owner()
            labels = {word for word in self.open_names
                      if forward_identity in owner.word_concepts(word)}
            from Spaces import _concept_alloc_of
            for word, identities in _concept_alloc_of(owner).word_forms.items():
                word = word.decode() if isinstance(word, bytes) else str(word)
                if forward_identity in identities:
                    labels.add(word.lower())
            if labels.intersection(self.open_names):
                return None
            return select_reference(choices, logits, forward_identity=forward_identity,
                                    individuals=individuals)
        def begin_word(fork, word, latches):
            self.word = int(word)
            return begin(fork, word, latches)
        def run_batch(model, *args, **kwargs):
            data = model.inputSpace.data
            self.texts = [getattr(data, kwargs['split']+'_input')[row] for row in kwargs['source_rows']]
            self.documents = [data.source_addresses[kwargs['split']][row]['document'] for row in kwargs['source_rows']]
            if self.question_budget is not None:
                model.attention_budget = self.question_budget if all(text.rstrip().endswith('?') for text in self.texts) else 0
            return batch(model, *args, **kwargs)
        def choose_operation(state, row_gate, **kwargs):
            self.current = state, row_gate, kwargs
            return choose(state, row_gate, **kwargs)
        def operation(x, **kwargs):
            previous, self.data = self.data, kwargs.get('reference_data')
            try:
                return forward(x, **kwargs)
            finally:
                self.data = previous
        def choose_logits(logits, **kwargs):
            result = select(logits, **kwargs)
            chosen, probabilities, legal = result[:3]
            fork = getattr(self.model, '_compose_fork', None)
            if self.data is None or fork is None:
                return result
            state, active, context = self.current
            scope = context['reference_scope']
            data = self.data
            from Spaces import _concept_alloc_of
            forms = _concept_alloc_of(self.model._concept_owner()).word_forms
            names = {}
            owner = self.model._concept_owner()
            for identity in owner.definitions.word_ids:
                for word in owner.definitions.description(identity)['forms']:
                    word = word.decode() if isinstance(word, bytes) else str(word)
                    for obj in (identity, *owner.word_concepts(word)):
                        names.setdefault(int(obj), set()).add(word.lower())
            for word, ids in forms.items():
                word = word.decode() if isinstance(word, bytes) else str(word)
                for identity in ids:
                    names.setdefault(int(identity), set()).add(word.lower())
            # The same forced references now name compose-bind locations.
            # Mention readings live with those locations in MTM.
            for occurrence, mention in owner.mtm.names.items():
                names.setdefault(mention.token[-1], set()).update(names.get(mention.reading, ()))
            nb, nu = len(data['binary_ops']), len(data['unary_ops'])
            for b in range(len(chosen)):
                if not bool(active[b]):
                    continue
                excluded = kwargs.get('masked_action')
                departure = excluded is not None and int(excluded[b]) >= 0
                depth = int(state[1][b])
                unary_position = min(depth, 2) - 1
                right = names.get(int(scope[b,0,1]), set())
                left = names.get(int(scope[b,1,1]), set()) if depth > 1 else set()
                identity_state = context.get('identity_state')
                bare = True
                if identity_state is not None:
                    identity_stack, journal = identity_state[2], identity_state[4]
                    right = right | names.get(int(identity_stack[b, 0, 4]), set())
                    if depth > 1:
                        left = left | names.get(int(identity_stack[b, 1, 4]), set())
                    # Transparent surface wrappers can drop scope flags. The
                    # overlay's producing action still distinguishes a bare
                    # pushed word from an already bound NP, VP or sentence.
                    bare = int(identity_stack[b, 0, 5]) == 0
                last = int(self.model.inputSpace._word_active_mask[b].nonzero()[-1])
                question = self.texts[b].rstrip().endswith('?')
                operation = None
                if identity_state is not None and right.intersection(self.reference_orders) and bare:
                    candidates = [i for i, op in enumerate(data['unary_ops'])
                        if language._compose_unary_rules[op].method_name == 'bind'
                        and bool(data['unary_valid'][b, unary_position, i])]
                    known = any(names.get(int(data['unary_refs'][b, unary_position, i, 0]), set()).intersection(right)
                                for i in candidates)
                    # Preserve the fixture's former mint/reference/open choices
                    # at their explicit grammatical operations.
                    operation = 'bind' if known or right.intersection(self.open_names) else 'lower_unary'
                elif depth > 1 and any(word.isspace() for word in left):
                    # Zero attention presents the whitespace units too.
                    # Force their ordinary surface attachment; leaving them
                    # on the stack would fabricate extra equality operands.
                    operation = 'surface'
                elif depth > 1 and 'the' in left:
                    operation = 'definite'
                elif (right.intersection(self.generic_subjects) and 'plus' in self.texts[b]
                      and not (int(scope[b,0,0]) & 2)):
                    operation = 'generic'
                elif depth > 1 and 'plus' in left and not (int(scope[b,1,0]) & 8) and not any(word.isspace() for word in right):
                    # The selected VP closes when its object arrives, before
                    # a following copula can be reduced into that object.
                    operation = 'verb'
                elif depth > 1 and 'plus' in right and (int(scope[b,0,0]) & 8) and not left.intersection({'is', 'plus'}):
                    operation = 'lift'
                elif self.word >= last and depth > 1:
                    if right.intersection({'.','?'}): operation = 'verb'
                    elif 'is' in left: operation = 'surface'
                    elif 'plus' in left: operation = 'verb'
                    elif 'plus' in right: operation = 'lift'
                    else: operation = 'equal'
                elif self.word >= last and depth == 1 and question:
                    prior = context.get('previous_unary')
                    ask = next(i for i,r in enumerate(language._compose_unary_rules) if r.method_name=='ask')
                    if prior is None or int(prior[b,0]) != ask:
                        operation = 'ask'
                allowed = torch.isfinite(logits[b])
                if departure:
                    allowed = allowed.clone()
                    allowed[int(excluded[b])] = False
                    eligible = kwargs.get('departure_eligible')
                    if eligible is not None:
                        allowed &= eligible[b]
                if operation is None:
                    candidates = [len(logits[b])-1] if bool(allowed[-1]) else []
                else:
                    unary = operation in ('ask', 'generic', 'bind', 'lower_unary')
                    rules = language._compose_unary_rules if unary else language._compose_binary_rules
                    ops = data['unary_ops' if unary else 'binary_ops']
                    refs = data['unary_refs' if unary else 'binary_refs'][b,unary_position if unary else 0]
                    offset = nb + unary_position * nu if unary else 0
                    def name(rule):
                        return 'definite' if getattr(rule,'determiner_mode',None)=='bind' else rule.method_name
                    candidates = [offset+i for i,op in enumerate(ops)
                                  if name(rules[op]) == operation and bool(allowed[offset+i])]
                    def rank(action):
                        identities = refs[action-offset].tolist()
                        score = 0
                        for role, labels in enumerate((left,right) if not unary else (right,)):
                            identity = identities[role]
                            position = 0 if unary else 1-role
                            # A completed phrase may retain its head's name.
                            # Rebinding that whole phrase to the bare head
                            # would erase its operands (four plus one -> four).
                            bare = operation == 'bind' or not (int(scope[b,position,0]) & (1 | 4 | 8))
                            if not bare:
                                score += 100 * (identity == -1)
                                continue
                            if self.preferred_references is not None:
                                preferred = self.preferred_references(b, labels)
                                if preferred is not None and operation == 'bind':
                                    token = owner.mtm.token_of(('ltm', resolver.store.address_domain, preferred))
                                    if token is not None:
                                        preferred = token[-1]
                                score += 1000 * (preferred is not None and identity == preferred)
                            if identity == (-1 if operation != 'bind' else 0): score += 1  # own object without a retained bond
                            if names.get(identity,set()).intersection(labels):
                                score += 3
                                store = self.model.symbolSpace.ltm_store
                                index = store.index_of_row(identity)
                                if index is not None:
                                    # The retired refs column was unused on
                                    # occurrences; its all--1 branch was +6.
                                    score += 6
                            if labels.intersection(self.open_names): score += 10 * (identity == 0)
                            for source, target in self.named_bindings.items():
                                if source in labels and target in names.get(identity, set()):
                                    score += 30
                            if 'answer' in labels:
                                slot = self.model.symbolSpace.ltm_store.slot_of_reference(identity)
                                if slot is not None: score += 20
                        return score
                    candidates.sort(key=rank, reverse=True)
                if candidates:
                    chosen[b] = candidates[0]
                    self.records.append(dict(row=b,word=self.word,operation=operation,
                        action=int(chosen[b]),forced_departure=departure,
                        scope=scope[b,:depth].tolist(),
                        left=sorted(left),right=sorted(right)))
                legal[b] = bool(allowed[chosen[b]])
            return result
        for owner,name,value in ((SentenceFork,'begin_word',begin_word),
                                 (BasicModel,'runBatch',run_batch),
                                 (language,'choose_operation',choose_operation),
                                 (layer,'forward',operation), (layer,'select_logits',choose_logits),
                                 (resolver,'select',reference_choice),
                                 (owner.interpret,'lookup_word',lookup_word)):
            self.stack.enter_context(patch.object(owner,name,value))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)
