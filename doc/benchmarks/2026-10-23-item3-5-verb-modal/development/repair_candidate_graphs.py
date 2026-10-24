from pathlib import Path
p=Path('bin/Language.py');s=p.read_text();old='''        binary = self._stacked_reduced(x, grammar_context, reference_data, operand_activations)
        unary = self._stacked_applied(x, grammar_context, reference_data)
''';new='''        # Scoring already reads detached candidates. In eager paired grammar
        # retain a graph only for the selected face, with its original live
        # operands. No bind value or selected operator loses its pullback.
        selected_grad = (not torch.compiler.is_compiling() and torch.is_grad_enabled()
            and grammar_context is not None and N <= 2
            and any(getattr(getattr(op, 'gl', op), 'meaning_layout', None) is not None
                    for op in self.ops))
        with torch.set_grad_enabled(torch.is_grad_enabled() and not selected_grad):
            binary = self._stacked_reduced(x, grammar_context, reference_data, operand_activations)
            unary = self._stacked_applied(x, grammar_context, reference_data)
''';assert s.count(old)==1;s=s.replace(old,new)
old='''        chosen = candidates.gather(1, action[:, None, None].expand(B, 1, D)).squeeze(1)
        probability = probabilities.gather(1, action[:, None])''';new='''        chosen = candidates.gather(1, action[:, None, None].expand(B, 1, D)).squeeze(1)
        if selected_grad:
            chosen = self._selected_grammar_candidate(x, chosen, kind, local_op, position,
                action, nb, r_reduce, r_apply, grammar_context, reference_data)
        probability = probabilities.gather(1, action[:, None])''';assert s.count(old)==1;s=s.replace(old,new)
needle='''    def _stacked_reduced(self, x, grammar_context=None, reference_data=None, activations=None):'''
helper='''    def _selected_grammar_candidate(self, x, chosen, kind, local, position,
                                    action, nb, r_reduce, r_apply, context, references):
        """Re-evaluate an eager selected face on the same live operands.

        The first evaluation supplies unchanged choices and probabilities.
        Its candidate values were already detached by the score function;
        only this selected numerical face participates in pathwise credit.
        Compiled composition retains its existing tensor-only evaluation.
        """
        batch = torch.arange(len(x), device=x.device)
        left = x[batch, position][:, None]
        right = x[batch, (position+1).clamp_max(x.shape[1]-1)][:, None]
        unary = left
        if references is not None:
            if r_reduce:
                column = action.remainder(r_reduce)
                at = position.clamp_max(references['left'].shape[1]-1)
                left = references['left'][batch, at, column][:, None]
                right = references['right'][batch, at, column][:, None]
            if r_apply:
                column = (action-nb).remainder(r_apply)
                unary = references['unary'][batch, position, column][:, None]
        for index, op in enumerate(self.ops):
            selected = (kind == 1) & (local == index)
            if not bool(selected.any()):
                continue
            value = (op.forward_with_grammar_context(left, right, x, context=context)
                     if hasattr(op, 'forward_with_grammar_context') else
                     op.forward_with_context(left, right, x) if hasattr(op, 'forward_with_context') else op(left, right))
            chosen = torch.where(selected[:, None], value[:, 0], chosen)
        for index, op in enumerate(self.unary_ops):
            selected = (kind == 2) & (local == index)
            if bool(selected.any()):
                value = self._apply_op(op, unary, grammar_context=context)
                chosen = torch.where(selected[:, None], value[:, 0], chosen)
        return chosen

''';assert s.count(needle)==1;s=s.replace(needle,helper+needle);p.write_text(s)
p=Path('bin/MeaningKernels.py');s=p.read_text().replace("q = self.groups(torch.zeros(value.shape[:-1], dtype=torch.long, device=value.device))", "q = self.groups(torch.zeros((), dtype=torch.long, device=value.device))")
old='''        if not self.modifier and object_value is None and group is None:
            w = self.spectrum(verb)''';new='''        # Unknown groups and absent objects have exactly the identity basis.
        # Eager evaluation need not build a dense matrix for that masked path.
        # The compiled path retains the tensor-only general expression.
        neutral_basis = object_value is None and group is None
        if not torch.compiler.is_compiling() and not self.modifier:
            neutral_basis = ((object_value is None or
                (object_present is not None and not bool(object_present.any())))
                and (group is None or not bool((group >= 0).any())))
        if not self.modifier and neutral_basis:
            w = self.spectrum(verb)''';assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
p=Path('test/test_sentence_end_state.py');s=p.read_text();old="""        retained_roles = (store.row(row)['meaning'].roles if isinstance(row, tuple)
                          else store.slots[row])""";new="""        retained_roles = store.row(row)['meaning'].roles""";assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
