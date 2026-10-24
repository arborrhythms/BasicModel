import json,tempfile
from pathlib import Path
import torch,pytest
from reading_fixtures import use_eager_reading
from test_tied_reconstruction_migration import _model,_run,_explicit_terminal_field
from SentenceUnderstanding import readback_scores
with pytest.MonkeyPatch.context() as p, tempfile.TemporaryDirectory() as d:
 use_eager_reading(p)
 m=_model(Path(d));out=_run(m,['aa bb cc']);_explicit_terminal_field(m)
 bank,valid=m._reconstruction_basis_snapshot(m._tensor_pushed_ideas)
 report={'pushed_norm':m._tensor_pushed_ideas.norm(dim=-1).tolist(),'bank_valid':valid.tolist(),'bank_norm':bank.norm(dim=-1).tolist(),'end_depth':m._tensor_final_end_depth.tolist(),'roots_depth':m._tensor_sentence_roots_depth.tolist(),'leaf_mask':m.inputSpace._ar_grammar_leaf_mask.tolist(),'owners':m.inputSpace._ar_concept_lookup_sentence_ids.tolist(),'ids':m.inputSpace._packed_sentence_ids.tolist(),'active':m.inputSpace._word_active_mask.tolist()}
 _,codes,_,_=m._snapshot_tables(m._tensor_pushed_ideas)
 report['codes_norm']=codes.norm(dim=-1).tolist()
 report['end_scores']=[readback_scores(m._tensor_final_end_slots[:,i],codes,valid.to(codes)).max(-1).values.tolist() for i in range(3)]
 dec=m._decode_conceptual_sentence(m._stm_single_S,m._tensor_final_end_slots,m._tensor_final_end_depth,codes,valid,torch.ones_like(valid,dtype=codes.dtype))
 report['decode']=[x.tolist() if torch.is_tensor(x) and x.numel()<500 else list(x.shape) if torch.is_tensor(x) else str(x) for x in dec]
 r=m._reconstruct_sentences(m._stm_single_S,m._tensor_pushed_ideas,m._tensor_sentence_roots_live,m._tensor_sentence_roots_depth,m._tensor_final_end_slots,m._tensor_final_end_depth)
 report['recovered_norm']=r[0].norm(dim=-1).tolist()
 report['root_norm']=m._tensor_sentence_roots_live.norm(dim=-1).tolist()
 Path('output/item3-5-verb-modal-20261023/development/owned-probe.json').write_text(json.dumps(report,indent=2))
 m.End();m.symbolSpace.soft_reset()
