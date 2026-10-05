def _output_generate_walk(self, event, budget, stamped_events=False,
                          targets=None, sample_actions=False,
                          basis=None, basis_valid=None, candidate_limit=16,
                          *, basis_priming=None, return_trace=False,
                          exploit_actions=None, departure=None,
                          return_candidates=False, case_bank=None,
                          initial_depth=None, require_symbols=True):
    """The shared conceptual decoder. No compose actions enter this walk.

    Hard generate choices determine the stack topology. Its numerical
    transition uses the usual straight-through softmax over candidate
    transitions, so reconstruction teaches the chooser as well as the
    inverse search. Output's restricted backward trains only its reader.
    """
    del targets, stamped_events
    language = self.languageSpace
    B, N, D = event.shape
    T = int(budget)
    R2, R1 = len(language._generate_binary_ops), len(language._generate_unary_ops)
    stop = R2 + R1
    ar = torch.arange(B, device=event.device)
    n_live0 = (event.abs().amax(-1).ne(0).sum(-1)
               if initial_depth is None else initial_depth)
    if exploit_actions is not None:
        # A preceding differentiable while_loop can expose symbolic
        # journal strides. Materialize its fixed-shape trace before it
        # becomes an invariant input of the second compiled loop.
        exploit_actions = exploit_actions.clone(memory_format=torch.contiguous_format)
    draws = torch.rand(B, T, device=event.device) if sample_actions else None
    departure_draw = (torch.rand(B, 1, device=event.device, dtype=event.dtype)
                      if exploit_actions is not None else None)

    def cond(t, stack, depth, emitted, count, actions, alternatives):
        return (t < T) & (depth > 0).any()

    def body(t, stack, depth, emitted, count, actions, alternatives):
        pos = (depth - 1).clamp(0, N - 1)
        parent = stack[ar, pos]
        live = depth > 0
        logits = language.generate_policy_logits(parent)
        # One candidate numerical transition per generate face. The hard
        # branch remains exact; soft probabilities supply its derivative.
        lefts, rights, unavailable = [], [], []
        for i in range(R2):
            index = torch.full_like(depth, i)
            if basis is None and require_symbols:
                left, right, missing = parent, parent, live
            else:
                left, right, missing = language.reverse_binary_step(
                    parent, index, live, ops=language._generate_binary_ops,
                    basis=basis if require_symbols else None,
                    basis_valid=basis_valid if require_symbols else None, basis_priming=basis_priming,
                    candidate_limit=candidate_limit, free=require_symbols, return_status=True, case_bank=case_bank)
            lefts.append(left); rights.append(right); unavailable.append(missing)
        for i in range(R1):
            value, missing = language.generate_unary_step(
                parent, torch.full_like(depth, i), live, return_status=True)
            lefts.append(value); rights.append(parent); unavailable.append(missing)
        lefts.append(parent); rights.append(parent); unavailable.append(torch.zeros_like(live))
        from Language import LanguageSpace
        available = ~torch.stack(unavailable, 1)
        # Native numerical answers realize through the shared inverse
        # chain; their values need not be an admitted word. Lexical
        # reconstruction retains its supported-pair/one-code eligibility.
        legal = (LanguageSpace.decoder_eligibility(parent, lefts, rights,
            available, language._generate_binary_ops,
            basis, basis_valid, case_bank=case_bank) if require_symbols else available)
        is_binary = torch.arange(stop + 1, device=event.device) < R2
        legal = legal & (~is_binary[None] | (depth < N)[:, None])
        if not torch.compiler.is_compiling() and _objective_probe.in_batch and getattr(self, "_sentence_training", False):
            _objective_probe.decoder_margin.capture(self, logits, parent, legal, round=t,
                explore=exploit_actions is not None, live=live,
                metadata=dict(epoch=_objective_probe.epoch, batch=_objective_probe.training_batches,
                              trial=getattr(self, "_sentence_trial", None)))
        logits = logits.masked_fill(~legal, -torch.inf)
        # An unreadable or capacity-blocked top remains pending. A finite
        # no-op distribution avoids NaNs but does not admit an emit.
        ready = legal.any(-1)
        logits = torch.where(ready[:, None], logits, torch.zeros_like(logits))
        choice = language.choose_generate(logits)
        if sample_actions:
            draw = draws.gather(1, t.reshape(1, 1).expand(B, 1))
            choice = (draw >= logits.detach().softmax(-1).cumsum(-1)).sum(-1).clamp_max(stop)
        if exploit_actions is not None:
            prior = exploit_actions.gather(1, t.reshape(1, 1).expand(B, 1)).reshape(B)
            excluded = F.one_hot(prior.clamp_min(0), stop + 1).bool()
            other_logits = logits.masked_fill(excluded, -torch.inf)
            from Language import sample_eligible_logits
            alternate = sample_eligible_logits(other_logits, departure_draw)
            choice = torch.where((t == departure) & legal.sum(-1).gt(1), alternate, choice)
            choice = torch.where((t < departure) & (prior >= 0), prior, choice)
        options = torch.stack(lefts, 1)
        others = torch.stack(rights, 1)
        probability = logits.softmax(-1)
        selected = F.one_hot(choice, stop + 1).to(parent)
        mixture = selected + (probability - probability.detach())
        left = (options * mixture[..., None]).sum(1)
        right = (others * mixture[..., None]).sum(1)
        missing = ~legal.gather(1, choice[:, None]).reshape(B)
        binary = live & (choice < R2) & (depth < N) & ~missing
        unary = live & (choice >= R2) & (choice < stop) & ~missing
        pop = live & (choice == stop) & ~missing
        rewritten = torch.where((binary | unary)[:, None], left, parent)
        stack1 = stack.clone()
        stack1[ar, pos] = torch.where(pop[:, None], 0., rewritten)
        above = (pos + 1).clamp_max(N - 1)
        stack1[ar, above] = torch.where(binary[:, None], right, stack1[ar, above])
        slot = (T - 1 - count).clamp(0, T - 1)
        emitted1 = emitted.clone()
        # At STOP, left is exactly the parent, with the chooser's
        # alternative numerical transitions retained for backward.
        emitted1[ar, slot] = torch.where(pop[:, None], left, emitted1[ar, slot])
        actions1 = actions.scatter(1, t.reshape(1, 1).expand(B, 1),
                                   torch.where(live & ready, choice, -1)[:, None])
        alternatives1 = alternatives.scatter(1, t.reshape(1, 1).expand(B, 1),
                                              (live & legal.sum(-1).gt(1))[:, None])
        return (t + 1, stack1, depth + binary.long() - pop.long(),
                emitted1, count + pop.long(), actions1, alternatives1)

    zeros = torch.zeros(B, device=event.device, dtype=torch.long)
    _, stack, depth, emitted, count, actions, alternatives = _reconstruction_while_loop(
        cond, body, _carries_with_grad((zeros.new_zeros(()), event.clone(), n_live0,
            event.new_zeros(B, T, D), zeros, zeros.new_full((B, T), -1),
            torch.zeros(B, T, device=event.device, dtype=torch.bool))))
    index = (torch.arange(T, device=event.device)[None] + (T-count)[:, None]).clamp_max(T-1)
    out = emitted.gather(1, index[..., None].expand(B, T, D))
    keep = torch.arange(T, device=event.device)[None] < count[:, None]
    out = torch.where(keep[..., None], out, 0.)
    result = (out, count, depth > 0, event.new_zeros(B))
    if return_candidates:
        return (*result, actions, alternatives)
    return (*result, actions) if return_trace else result
