import json
import torch
from Models import BasicModel
from test_space_equiv_selfcheck import test_identity_candidate_passes as run_original


def test_identity_candidate_trace_diagnostic(monkeypatch):
    original = BasicModel._sentence_observation
    def observe(self, state, sid, active, **kwargs):
        try:
            return original(self, state, sid, active, **kwargs)
        except ValueError:
            stm, lang, _ = state
            trace = self._reconstruction_stack()
            print(json.dumps(dict(sid=sid, depth=stm[1].tolist(), cap=self.conceptualSpace.stm.capacity,
                active=self.inputSpace._word_active_mask.tolist(),
                sentence_ids=self.inputSpace._packed_sentence_ids.tolist(),
                end_mask=self.inputSpace._packed_sentence_end_mask.tolist(),
                binary_map=trace.rule_map(2).tolist(), compose_map=self.languageSpace._cs_binary_rule_ids.tolist(),
                unary_map=trace.rule_map(1).tolist(), compose_unary=self.languageSpace._cs_unary_rule_ids.tolist(),
                actions=[[(i,int(lang[4][b,i]),int(lang[5][b,i]),int(lang[18][b,i]))
                    for i in (lang[6][b]).nonzero().flatten()] for b in range(len(stm[1]))])))
            raise
    monkeypatch.setattr(BasicModel, '_sentence_observation', observe)
    run_original()
