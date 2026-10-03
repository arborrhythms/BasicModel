"""Remove audited unreferenced helpers and four already retired skipped cases."""
from port_ledger import ROOT,HERE,record,remove_definitions
for filename,names in {
 'bin/Queries.py':['GrammaticalQueryRegistry','checked_query_declarations'],
 'bin/Language.py':['_active_percept_prototypes','_rotate_where'],
 'bin/embed.py':['_CBOWModule'],
}.items():
 for name,old in remove_definitions(filename,names).items():
  record(filename,name,old,[], 'Unreferenced implementation: no import, direct/dynamic/string/configuration/grammar or checkpoint-key consumer.', ['precursor-and-dead-code-use-audit.json','legacy-checkpoint-string-audit.json'])
p=ROOT/'bin/Models.py';s=p.read_text();line='        self._ar_valid_pos = None  # IR has no per-cursor axis.\n';assert s.count(line)==1;s=s.replace(line,'');p.write_text(s)
(HERE/'removed-model-stub.txt').write_text(line+'\nOnly consumer was the retired test_loss_mask_matches_stem_valid_mask, removed in item 5. Full original Models.py is in pre-trim-source.zip.\n')
for filename,names in {
 'test/test_explicit_ordered_stm.py':['test_stm_parses_explicit_np3_lift_to_s4','test_stm_parses_explicit_np4_lift_to_s5'],
 'test/test_input_word_cursor.py':['test_live_forward_whole_slab_byte_identical_and_cursor_wired'],
 'test/test_padded_rows_no_op.py':['test_loss_mask_matches_stem_valid_mask'],
}.items():
 for name,old in remove_definitions(filename,names).items():
  record(filename,name,old,[], 'Retired feature: existing test always skipped on the live architecture; deletion explicitly requested in trim item 5.', ['October 1 hand-off part 4 item 5','pre-trim-source.zip'])
print('Removed five unused definitions, the unused _ar_valid_pos stub, and four retired skipped tests.')
