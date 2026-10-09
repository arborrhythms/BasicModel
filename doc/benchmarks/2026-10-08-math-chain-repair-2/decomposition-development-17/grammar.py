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
                 preferred_references=None):
        self.model, self.open_names = model, set(open_names)
        self.named_bindings = dict(named_bindings or {})
        self.generic_subjects = set(generic_subjects)
        self.question_budget = question_budget
        self.preferred_references = preferred_references
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
            chosen, probabilities, legal = select(logits, **kwargs)
            fork = getattr(self.model, '_compose_fork', None)
            if self.data is None or fork is None:
                return chosen, probabilities, legal
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
            nb, nu = len(data['binary_ops']), len(data['unary_ops'])
            for b in range(len(chosen)):
                if not bool(active[b]):
                    continue
                excluded = kwargs.get('masked_action')
                departure = excluded is not None and int(excluded[b]) >= 0
                depth = int(state[1][b])
                right = names.get(int(scope[b,0,1]), set())
                left = names.get(int(scope[b,1,1]), set()) if depth > 1 else set()
                last = int(self.model.inputSpace._word_active_mask[b].nonzero()[-1])
                question = self.texts[b].rstrip().endswith('?')
                operation = None
                if depth > 1 and any(word.isspace() for word in left):
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
                    unary = operation in ('ask', 'generic')
                    rules = language._compose_unary_rules if unary else language._compose_binary_rules
                    ops = data['unary_ops' if unary else 'binary_ops']
                    refs = data['unary_refs' if unary else 'binary_refs'][b,0]
                    offset = nb if unary else 0
                    def name(rule):
                        return 'definite' if getattr(rule,'determiner_mode',None)=='bind' else rule.method_name
                    candidates = [offset+i for i,op in enumerate(ops)
                                  if name(rules[op]) == operation and bool(allowed[offset+i])]
                    def rank(action):
                        identities = refs[action-offset].tolist()
                        score = 0
                        for role, labels in enumerate((left,right) if not unary else (right,)):
                            identity = identities[role]
                            if self.preferred_references is not None:
                                preferred = self.preferred_references(b, labels)
                                score += 1000 * (preferred is not None and identity == preferred)
                            if identity == -1: score += 1  # mint when there is no retained bond
                            if names.get(identity,set()).intersection(labels):
                                score += 3
                                store = self.model.symbolSpace.ltm_store
                                index = store.index_of_row(identity)
                                if index is not None:
                                    cached = store.refs[index]
                                    score += 6 if bool((cached == -1).all()) else -20
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
                        left=sorted(left),right=sorted(right)))
                legal[b] = bool(allowed[chosen[b]])
            return chosen, probabilities, legal
        for owner,name,value in ((SentenceFork,'begin_word',begin_word),
                                 (BasicModel,'runBatch',run_batch),
                                 (language,'choose_operation',choose_operation),
                                 (layer,'forward',operation), (layer,'select_logits',choose_logits)):
            self.stack.enter_context(patch.object(owner,name,value))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)
