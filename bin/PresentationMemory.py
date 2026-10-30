"""Rewrite document covers and discard epoch experience, retaining concepts."""
import torch


# Returned readings and diagnostic records can own the same occurrence handles
# as the active discourse. None is a parameter or a persistent concept record.
_EMPTY_DICTS = (
    '_document_covers', '_sentence_fields', '_sentence_understandings',
    '_what_pending', '_what_root_symbols', '_what_referents',
    '_reference_close_audit', '_costed_thought_outputs', '_thought_ordinals',
)
_EMPTY_LISTS = (
    '_what_policy_records', '_what_episode_steps', '_verb_teacher_records',
    '_modal_teacher_records', '_sentence_credit_audits', '_sentence_trial_costs',
    '_sentence_winners', '_sentence_reconstructions',
)
_EMPTY_TUPLES = ('_thought_document_ends', '_last_what_answers',
                 '_last_what_actual', '_active_what_questions', '_sentence_answer_questions',
                 '_last_closing_thoughts')
_ABSENT = (
    '_last_understanding', '_last_sentence_understanding', '_sentence_field',
    '_sentence_symbol_field', '_last_sentence_field', '_last_field_cost',
    '_sentence_primed_bank', '_sentence_inverse_candidates', '_last_selected_thought',
    '_last_thought_comparison', '_last_answer_construction', '_last_output_comparison',
    '_last_output_walk_trace', '_last_decoder_comparison', '_last_decoder_trace',
    '_answer_attention_obs', '_thought_walk', '_compose_fork',
    '_sentence_answer_cost', '_sentence_answer_predictions', '_sentence_answer_raw_cost',
    '_last_compose_score_function', '_sentence_journal_columns', '_query_ready_rows',
    '_sentence_supplied_answers', '_sentence_comparison_registry',
    '_sentence_reader_observation', '_last_sentence_credit', '_last_what_thinking',
    '_recall_brackets',
)


def begin(model):
    """Begin a document presentation; its cover is rewritten on first read."""
    if getattr(model, '_sentence_run_active', False) or getattr(model, '_open_sentence_slot', None) is not None:
        raise RuntimeError('a presentation cannot begin inside a live sentence')
    model._document_covers = {}


@torch.no_grad()
def epoch(model):
    """Discard epoch experience after backward, preserving learned concepts."""
    begin(model)
    percept = getattr(model,'percept_expectation',None)
    if percept is not None:
        percept.clear_context()
    symbol = getattr(model, 'symbolSpace', None)
    store = getattr(symbol, 'ltm_store', None)
    owners = list(getattr(model, 'conceptualSpaces', ()) or ())
    owner = getattr(model, 'conceptualSpace', None)
    if owner is not None and all(owner is not item for item in owners):
        owners.append(owner)
    # Promotion derives an individual's order from its retained occurrence.
    # Capture that witnessed result before retiring the epoch's experience,
    # as consolidation does when it retains a learned object without its rows.
    witnesses, witnessed_allocators = [], set()
    for item in owners:
        alloc = getattr(item, '_concept_allocator', None)
        if alloc is not None and id(alloc) not in witnessed_allocators:
            witnessed_allocators.add(id(alloc))
            for cid in alloc.individuals:
                refs = [ref for role, ref in alloc.store_of(cid).constituents(cid)
                        if role == 'part' and isinstance(ref, tuple) and ref[:1] == ('occ',)]
                if refs:
                    witnesses.append((alloc, cid, alloc.order_of(cid), refs))
    for item in owners:
        item.mtm.restore(None)
    removed, removed_parts = set(), set()
    if store is not None:
        before = set(store.address_keys[:len(store)].tolist())
        # clear_origin preserves the dependency closure of given knowledge.
        # Without given rows, reset also clears every unused row and index.
        if bool((store.origin[:len(store)] != store.ORIGIN_CONVERSATION).any()):
            store.clear_origin(store.ORIGIN_CONVERSATION)
        else:
            store.reset()
        store.clear_recurrence()
        removed = before - set(store.address_keys[:len(store)].tolist())
        parts = store.parts
        from LTM1 import OCCURRENCE, RELATION
        invalid = {(OCCURRENCE, key) for key in removed}
        while True:
            found = {row for row in parts.indices().tolist()
                     if row not in removed_parts and any(ref in invalid for ref in parts.references(row))}
            if not found:
                break
            removed_parts.update(found)
            invalid.update((RELATION, row) for row in found)
        if removed_parts:
            parts.retain([row for row in parts.indices().tolist() if row not in removed_parts])
        for name in ('forget_passes', 'forget_sentences', 'forget_last_sentence',
                     'forget_cutoff', 'forget_pending'):
            getattr(store, name).zero_()
        for name in ('_slot_index_source', '_slot_index', '_forward_index_source',
                     '_forward_postings', '_forward_by_address', 'last_forgetting_report',
                     'last_part_forgetting_report'):
            store.__dict__.pop(name, None)
        store.rebuild_addresses()
    for alloc, cid, order, refs in witnesses:
        expired = [ref for ref in refs if ref[1] in removed]
        if expired:
            if order < 0:
                raise RuntimeError('a promoted identity has no witnessed order at the epoch boundary')
            alloc.reference_orders[cid] = order
            for ref in expired:
                alloc.store_of(cid).remove_constituent(cid, 'part', ref)
            alloc.settle(cid)
    seen = set()
    for item in owners:
        alloc = getattr(item, '_concept_allocator', None)
        if alloc is not None and id(alloc) not in seen:
            seen.add(id(alloc))
            alloc.testimony_seen.clear()
            alloc.testimony_current.clear()
            alloc.relate_idx = {key: value for key, value in alloc.relate_idx.items()
                                if value not in removed_parts}
        components = getattr(item, 'components', None)
        if components is not None:
            components._innovations.clear()
            components._chart = None
            components.verbs._pending.clear()
            components.verbs._witnesses.clear()
            resolver = components.reference_close
            resolver.__dict__.pop('_mentions', None)
            resolver.last_audit = ()
            for name in ('last_verb_teacher', 'last_modal_teacher'):
                resolver.__dict__.pop(name, None)
        derived = getattr(getattr(item, 'similarity_codebook', None), 'mereology', None)
        # Knowing's eager carrier can retain an occurrence after its thought
        # history has gone. Sentence STM itself is initialized by the driver.
        carrier = getattr(item, 'subspace', None)
        if carrier is not None:
            object.__setattr__(carrier, '_thought_occurrence', None)
    for name in _EMPTY_DICTS:
        setattr(model, name, {})
    for name in _EMPTY_LISTS:
        setattr(model, name, [])
    for name in _EMPTY_TUPLES:
        setattr(model, name, ())
    for name in _ABSENT:
        setattr(model, name, None)
    for name in ('_forget_document_boundary', '_forget_all_streams_closed', '_forget_training'):
        model.__dict__.pop(name, None)
    model._clear_recall_history()
    for name in ('expectation', 'what_memory'):
        memory = getattr(symbol, name, None)
        if memory is not None:
            memory.reset()
            if name == 'what_memory':
                memory.__dict__.pop('_address_sources', None)
            else:
                memory.set_symbol_bank(memory._bank_keys[:0], identities=memory._bank_ids[:0],
                    orders=memory._bank_orders[:0], addresses=memory._bank_addresses[:0])
    language = getattr(model, 'languageSpace', None)
    if language is not None:
        for name in ('verb_action', 'modifier_action'):
            action = getattr(language, name)()
            if action is not None:
                action.groups.witnesses.clear()
                action.groups.reports.clear()
        for name in ('_reference_bank', '_compose_primed_bank', '_last_derivation',
                     '_what_context', '_what_ltm_context'):
            setattr(language, name, None)
    truth = getattr(symbol, 'truth_layer', None)
    if truth is not None:
        truth.sync_from_ltm()
    return dict(removed=len(removed), relations_removed=len(removed_parts),
                retained=0 if store is None else len(store))
