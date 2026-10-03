"""Retire uncalled legacy credit/scalar APIs, preserving live owner behavior."""
import ast,json,textwrap
from port_ledger import ROOT,HERE,definitions,record,remove_definitions
E=['remaining-legacy-entrypoint-uses.txt','legacy-checkpoint-string-audit.json']
changes=[]
def change(file,fn):
 p=ROOT/file;old=p.read_text();new=fn(old);ast.parse(new);p.write_text(new)
 changes.append(dict(file=file,old=old,new=new))
def retire(file,names,reason):
 for name,body in remove_definitions(file,names).items():
  record(file,name,body,[],reason,E)
def port(file,name,fn,reason):
 p=ROOT/file;s=p.read_text();old=definitions(s)[name];new=fn(old);assert new!=old
 p.write_text(s.replace(old,new,1));ast.parse(p.read_text())
 record(file,name,old,[(file,name)],reason,E)
# The scalar adder has no production or configuration caller. Keep the vector
# buffer consumed by runEpoch / fineweb / bucket benchmarks.
port('bin/Spaces.py','SubSpace::add_word',lambda old:'''    def add_word(self, batch, vector, rule, order=0,
                 leaf1=-1, leaf2=-1, leaf3=-1, pos=0):
        """Scatter word entries for active tensor row IDs into the tick buffer."""
        self._add_word_vec(batch, vector, rule, order=order,
                           leaf1=leaf1, leaf2=leaf2, leaf3=leaf3, pos=pos)
''','Remove the uncalled scalar list overload; retain the vector tick buffer and its production flush callers.')
retire('test/test_basicmodel.py',['TestSubspaceWords::test_add_word_start_state','TestSubspaceWords::test_add_multiple_words','TestSubspaceWords::test_add_word_validates'],'These cases exercise only the retired scalar word-adder overload; WordEncoding validation tests remain.')
retire('test/test_word_buffer_flush.py',['test_legacy_scalar_path_unaffected_by_buffer'],'Only the retired scalar word-adder overload is exercised.')
def buffer_port(old):
 old=old.replace('Scalar add_word and vector add_word produce identical entries.','The vector buffer reproduces the canonical WordEncoding tuples.')
 old=old.replace('through both APIs and compare the materialized ``self.word`` list','through WordEncoding and compare the materialized ``self.word`` list')
 old=old.replace('    for (b, v, r, o, l1, l2, l3) in seq:\n        ref_sub.add_word(b, v, r, order=o, leaf1=l1, leaf2=l2, leaf3=l3)\n    ref_words = list(ref_sub.word)','    ref_words = [ref_sub.wordEncoding.encode(b, v, r, o, l1, l2, l3)\n                 for b, v, r, o, l1, l2, l3 in seq]')
 return old.replace('# Reference: scalar path.','# Reference: canonical tuple encoding.')
port('test/test_word_buffer_flush.py','test_scalar_and_vector_overloads_agree',buffer_port,'Use the independent canonical tuple encoder as the reference after scalar API retirement; sequence and exact equality assertion unchanged.')
# No production call, dynamic name, configuration or checkpoint opens a legacy
# credit reservation. Ordinary thought episodes own this live credit now.
retire('bin/Layers.py',['WhatInteractionMemory::begin_what_episode','WhatInteractionMemory::_live_what_value'],'Uncalled legacy credit-only API retired; begin_thought_episode owns live credit.')
def detach_legacy(s):
 s=s.replace('        live = (self.detach_mode == "episode" and bi in self._episode_live)\n        snapshot = self._live_what_value if live else self._detach_what_value','        snapshot = self._detach_what_value')
 s=s.replace('        if live:\n            self._episode_live[bi].append(stored)\n','')
 a=s.index('                    if isinstance(slot, ThoughtRecord):',s.index('    def end_what_episode('));b=s.index('                    replaced += 1',a)
 s=s[:a]+'                    slot = slot.snapshot(detach=True)\n'+s[b:]
 s=s.replace('    legacy ``begin_what_episode``) LIVE on the autograd graph so the root answer','    ``begin_thought_episode``) LIVE on the autograd graph so the root answer')
 s=s.replace('# row -> live legacy/ordinary records','# row -> live ordinary thought records')
 s=s.replace('        detached at append unless the row is inside an ``episode``-mode\n        episode.','        detached at append. Live episode credit belongs to ordinary thought records.')
 return s
change('bin/Layers.py',detach_legacy)
retire('test/test_thought_legacy_boundaries.py',[
 'test_legacy_credit_begin_cannot_replace_an_active_ordinary_episode',
 'test_legacy_credit_reservation_can_end_before_forward_establishes_batch',
 'test_legacy_adapter_detaches_complete_meanings_at_the_credit_boundary',
 'test_checkpoint_snapshot_detaches_legacy_values_without_detaching_the_live_owner'],
 'Only the uncalled legacy credit reservation path is retired. Native thought credit/boundary cases remain in test_thought_history_boundaries.py and test_thought_review.py; legacy checkpoint replay and underflow cases remain.')
retire('test/test_what_episode_memory.py',['test_episode_mode_keeps_values_live_until_end'],'Legacy slot credit is retired; ordinary history credit remains exercised by test_thought_history_boundaries.py.')
port('test/test_what_episode_memory.py','test_slot_mode_detaches_at_append',lambda s:s.replace('    memory.begin_what_episode(0)\n',''),'The retained LTMSlot append path always detaches; remove the retired reservation setup, retaining every assertion.')
def owner_port(s):
 a=s.index('    memory.begin_what_episode(0)');b=s.index('    model.symbolSpace.ensure_microbatch',a)
 return s[:a]+'''    from Meaning import ConceptualMeaning
    x = torch.ones(3, 2, requires_grad=True)
    meaning = ConceptualMeaning(x, torch.ones(3, dtype=torch.bool))
    memory.begin_thought_episode(meaning, work_budget=2)
    stored = memory.thought_history()[0]
    assert stored.meaning.roles.requires_grad
    memory.finish_thought(meaning)
    assert memory.end_what_episode(0) == 2  # begin plus the explicit finish
    assert not memory.thought_history()[0].meaning.roles.requires_grad
'''+s[b:]
port('test/test_what_episode_memory.py','test_symbol_space_owns_memory_lifecycle',owner_port,'Port model-owned credit/lifecycle to ordinary begin/finish records. Exactly two records replace the one legacy complete slot; all ownership, gradient-detachment, resizing and row-reset subjects stay.')
retire('test/test_thought_checkpoint_credit.py',['test_checkpoint_question_prompt_is_detached_without_cutting_live_credit','test_episode_end_releases_question_and_trace_credit_together'],'Only legacy live-credit reservation is retired. Keep the legacy checkpoint prompt-detachment test and native history credit tests.')
port('test/test_thought_checkpoint_credit.py','_live_memory',lambda s:s.replace('    memory.begin_what_episode()\n',''),'Checkpoint fixture uses the retained durable legacy slot API without the retired live-credit reservation. Remaining checkpoint assertions unchanged.')
def reset_port(s):
 s=s.replace('    memory.begin_what_episode(0)\n    memory.append_what_slot(LTMSlot(input=torch.ones(4)))','    from Meaning import ConceptualMeaning\n    meaning = ConceptualMeaning(torch.ones(3, 4), torch.ones(3, dtype=torch.bool))\n    memory.begin_thought_episode(meaning, work_budget=2)')
 return s+'    assert memory.thought_history() == []\n'
port('test/test_expectation_review.py','test_provisioning_keeps_its_existing_hard_reset_of_what_episode',reset_port,'Port the setup to the current thought episode. Preserve reset assertions and additionally check the ordinary history is empty.')
# No in-use checkpoint carries the pre-step-7 serialized clause derivation.
# Generic older semantic end-state formats stay supported.
def clause_legacy(s):
 s=s.replace('        migrated_fingerprints = {}\n        discarded_derivation = False\n','')
 s=s.replace("            if version >= 4 and 'clause' in record:","            if 'clause' in record:")
 a=s.index("            legacy = record.get('clause') if version == 3 else None");b=s.index('            meaning = ConceptualMeaning(',a)
 s=s[:a]+'            stored_context = dict(context)\n'+s[b:]
 s=s.replace(' or legacy is not None) and not bool(self.metadata_required[i])',') and not bool(self.metadata_required[i])')
 a=s.index('            if legacy is None:\n                fingerprint_values =');b=s.index('            fingerprint = self.semantic_fingerprint',a)
 s=s[:a]+'            fingerprint_values = self._context_fingerprint(incoming[identifier], text, expectation, definition)\n'+s[b:]
 a=s.index('        for index, values in migrated_fingerprints.items():');b=s.index('        if getattr(self, "_leaf_index_missing", False):',a)
 return s[:a]+s[b:]
change('bin/Layers.py',clause_legacy)
retire('test/test_item7_end_state_storage.py',['test_legacy_checkpoint_drops_the_derivation_after_validating_its_binding'],'No in-use checkpoint contains a stored clause/program payload. Its special version-3 migration is removed; all current end-state and supported older semantic-format tests stay.')
(HERE/'legacy-credit-and-record-source-changes.json').write_text(json.dumps(changes,indent=2)+'\n')
print('Retired unused scalar append, legacy credit reservation, and old clause-program migration; preserved live owner/checkpoint behaviors.')
