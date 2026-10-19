#!/usr/bin/env python3
"""Native gain consumer measurements, with an explicit perfect-prior control."""
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch,pytest
from test_compiled_word_chunk import _tiny_canonical_model,_stage_fullgraph_tensor_peer
from What import What
import util, ThoughtClosing

def run():
    torch.set_num_threads(1)
    result=[]
    for strength in (0.,.5,1.):
        with tempfile.TemporaryDirectory() as td,pytest.MonkeyPatch.context() as patch:
            patch.setattr(util,'TheCompileBackend','none')
            model=_tiny_canonical_model(Path(td),patch,input_width=16,word_buckets='8',stm_capacity=3,
                chooser_depth=1,training_overrides={'intraLossWeight':0.,'symbolExpectationGain':strength,
                'rowExpectationGain':strength,'documentExpectationGain':strength},
                architecture_overrides={'ltmConsolidation':True,'conceptLayers':2})
            model.eval();model._tensor_peer_while_eager=True
            from test_reverse_traversal import _select_completed_binary_path
            _select_completed_binary_path(model)
            images=[];end_states=[]
            close=ThoughtClosing.close
            def capture(*args,**kwargs):
                reducer=model._stm_reducer()
                assert reducer._expectation_prime is None
                assert reducer._symbol_expectation_read is None
                images.append(kwargs['images'])
                end_states.append(tuple(args[1]))
                return close(*args,**kwargs)
            patch.setattr(ThoughtClosing,'close',capture)
            # This is a controlled perfect-prior mechanism check, not a claim
            # that the failed curriculum learned a predictor to confidence.
            predictor=model.symbolSpace.expectation._inter_predictor
            with torch.no_grad():
                for p in predictor.parameters():p.zero_()
                predictor.local.bias.copy_(torch.tensor([60.,-60.]))
                for repeat in range(2):
                    raw=_stage_fullgraph_tensor_peer(model,['cat','cat'])
                    with model._sentence_run():
                        outputs=model._forward_with_compiled_sentence_state(raw)
                    public=model._publish_compiled_sentence_state(outputs);model._end_step()
                    model.End();model.symbolSpace.soft_reset()
                understanding=model._capture_understanding(public)
                questions=(What.present(0),What.present(0))
                answer=model.resolveAnswer(understanding,questions)
                answer_idea=model._materialize_answer_idea(understanding,answer,questions)[0]
                field=end_states[-1][0];view=images[-1][0]
                store=model.symbolSpace.ltm_store
                index=store.index_of_row(field.row_id)
                pair=store.expectation_pair(index,gain=strength)
                item=dict(gain=strength,resolved=answer.resolved,stored_index=index,
                    raw=field.meaning.roles.tolist(),content=view.conceived.tolist(),gain_values=view.gain.tolist(),
                    expected=None if view.expected is None else view.expected.tolist(),
                    pair_present=pair is not None,answer=answer_idea.tolist(),
                    bank_min_distance=torch.cdist(view.observed[None],model.symbolSpace.expectation._bank_keys[None],compute_mode='donot_use_mm_for_euclid_dist').min(-1).values.tolist(),
                    comparison_observed=model.symbolSpace.expectation.last_expectation_comparison().observed.tolist(),
                    refs=field.meaning.role_refs, bank_ids=model.symbolSpace.expectation._bank_ids.tolist(),
                    bank_keys=model.symbolSpace.expectation._bank_keys.tolist(),
                    word_ids=model.inputSpace._ar_word_concept_ids.tolist(),
                    word_atoms=model.inputSpace._ar_word_object_atoms.tolist(),
                    prime_cleared=True,observation_count=model.symbolSpace.expectation.expectation_metrics()['observations'])
                torch.testing.assert_close(view.observed,field.meaning.roles)
                torch.testing.assert_close(store.meaning_of(index).roles,field.meaning.roles)
                torch.testing.assert_close(answer_idea[0],field.meaning.roles)
                assert answer.resolved and pair is not None
                result.append(item)
            model.End()
    return result

if __name__=='__main__':
    report=run()
    (Path(__file__).parent/'native_gain_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print([{k:r[k] for k in ('gain','resolved','pair_present','gain_values')} for r in report])
