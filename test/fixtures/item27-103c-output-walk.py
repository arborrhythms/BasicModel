# Frozen 103c66621c4c reference, used only by the exact native replay certificate.
def _output_generate_walk(self, event, budget, stamped_events=False,
                          targets=None, sample_actions=False,
                          basis=None, basis_valid=None, candidate_limit=16,
                          *, basis_priming=None, return_trace=False,
                          exploit_actions=None, departure=None,
                          return_candidates=False, case_bank=None,
                          initial_depth=None, require_symbols=True,
                          constituents=None, constituent_valid=None, constituent_families=None,
                          terminal_valid=None, whole_bank=None):
    """The shared conceptual decoder. No compose actions enter this walk.

    Hard generate choices determine the stack topology. Reconstruction
    teaches the policy separately from the compose derivation's structure;
    this free walk supplies no straight-through policy gradient.
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
    # while_loop lifts these snapshots as invariant inputs. Callers may
    # supply overlapping views (a primed row can also be a live whole),
    # but HOP inputs must have independent storage, including masks.
    def own(value):
        return value.clone(memory_format=torch.contiguous_format) if torch.is_tensor(value) else value
    basis, basis_valid, basis_priming = map(own, (basis, basis_valid, basis_priming))
    constituents, constituent_valid = map(own, (constituents, constituent_valid))
    terminal_valid, n_live0 = own(terminal_valid), own(n_live0)
    if case_bank is not None:
        case_bank = type(case_bank)(*(own(value) for value in case_bank))
    if whole_bank is not None:
        whole_bank = type(whole_bank)(*(own(value) for value in whole_bank))
    if exploit_actions is not None:
        # A preceding differentiable while_loop can expose symbolic
        # journal strides. Materialize its fixed-shape trace before it
        # becomes an invariant input of the second compiled loop.
        exploit_actions = exploit_actions.clone(memory_format=torch.contiguous_format)
    draws = torch.rand(B, T, device=event.device) if sample_actions else None
    departure_draw = (torch.rand(B, 1, device=event.device, dtype=event.dtype)
                      if exploit_actions is not None else None)
    # The first menu keeps the zero-gradient edges of rejected operators,
    # including their optimizer participation. Later menus use the same
    # catalogue and bank; only the selected inverse needs another graph.
    zero_menu = [None]
    # Host equality checks belong only to the explicitly eager CPU path.
    # Compiled/device walks keep their tensor-only body. A cache hit does
    # not skip a step, change a choice or reuse a selected inverse graph.
    import util
    from Generative import InverseMenuRead
    menu_read = (InverseMenuRead(language, N)
                 if not torch.compiler.is_compiling() and event.device.type == 'cpu'
                 and util.TheCompileBackend == 'none' else None)
    if menu_read is not None:
        from FormCentroids import prepare_inverse
        whole_bank = prepare_inverse(whole_bank)

    def cond(t, stack, depth, emitted, count, actions, alternatives, progressing):
        return (t < T) & (depth > 0).any() & progressing

    def body(t, stack, depth, emitted, count, actions, alternatives, progressing):
        pos = (depth - 1).clamp(0, N - 1)
        parent = stack[ar, pos]
        live = depth > 0
        logits = language.generate_policy_logits(parent)
        from Generative import inverse_menu, selected_inverse
        selected_graph = torch.is_grad_enabled() and not torch.compiler.is_compiling()
        first_graph = selected_graph and zero_menu[0] is None
        with torch.set_grad_enabled(torch.is_grad_enabled() and (not selected_graph or first_graph)):
            menu_options = dict(
                basis=basis, basis_valid=basis_valid, priming=basis_priming,
                limit=candidate_limit, case_bank=case_bank,
                constituents=constituents, constituent_valid=constituent_valid,
                constituent_families=constituent_families,
                require_symbols=require_symbols, terminal_valid=terminal_valid, whole_bank=whole_bank)
            options, others, legal = (inverse_menu(language, parent, live, **menu_options)
                if menu_read is None else menu_read.read(parent, live, **menu_options))
        if first_graph:
            zero_menu[0] = (options.sum() + others.sum()) * 0
        is_binary = torch.arange(stop + 1, device=event.device) < R2
        legal = legal & (~is_binary[None] | (depth < N)[:, None])
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
        selected = F.one_hot(choice, stop + 1).to(parent)
        left = (options * selected[..., None]).sum(1)
        right = (others * selected[..., None]).sum(1)
        missing = ~legal.gather(1, choice[:, None]).reshape(B)
        if selected_graph:
            left, right = selected_inverse(language, parent, choice, live & ~missing,
                basis=basis, basis_valid=basis_valid, priming=basis_priming,
                limit=candidate_limit, case_bank=case_bank, constituents=constituents,
                constituent_valid=constituent_valid, require_symbols=require_symbols, whole_bank=whole_bank)
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
        # At STOP, left is exactly the parent.
        emitted1[ar, slot] = torch.where(pop[:, None], left, emitted1[ar, slot])
        actions1 = actions.scatter(1, t.reshape(1, 1).expand(B, 1),
                                   torch.where(live & ready, choice, -1)[:, None])
        alternatives1 = alternatives.scatter(1, t.reshape(1, 1).expand(B, 1),
                                              (live & legal.sum(-1).gt(1))[:, None])
        return (t + 1, stack1, depth + binary.long() - pop.long(),
                emitted1, count + pop.long(), actions1, alternatives1,
                (live & ready).any())

    # With no legal move in any live row, the remaining trips would
    # repeat this exact state. Keep its pending depth and untouched
    # trace suffix; the random draws were allocated for the whole budget.
    zeros = torch.zeros(B, device=event.device, dtype=torch.long)
    _, stack, depth, emitted, count, actions, alternatives, _ = _reconstruction_while_loop(
        cond, body, _carries_with_grad((zeros.new_zeros(()), event.clone(), n_live0,
            event.new_zeros(B, T, D), zeros, zeros.new_full((B, T), -1),
            torch.zeros(B, T, device=event.device, dtype=torch.bool),
            torch.ones((), device=event.device, dtype=torch.bool))))
    index = (torch.arange(T, device=event.device)[None] + (T-count)[:, None]).clamp_max(T-1)
    out = emitted.gather(1, index[..., None].expand(B, T, D))
    keep = torch.arange(T, device=event.device)[None] < count[:, None]
    out = torch.where(keep[..., None], out, 0.)
    if zero_menu[0] is not None:
        out = out + zero_menu[0]
    result = (out, count, depth > 0, event.new_zeros(B))
    if return_candidates:
        return (*result, actions, alternatives)
    return (*result, actions) if return_trace else result
