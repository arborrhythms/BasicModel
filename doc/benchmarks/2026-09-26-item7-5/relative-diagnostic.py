"""Read-only instrumentation of the unchanged existing relative-depth campaign."""
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
import torch
import Models
from Language import TheGrammar
from test_thinking_kernel import TestDepth3RelativeEndState
old = Models.BasicModel._sentence_relative_mask
calls = []
def observed(self, B, device=None):
    result = old(self, B, device=device)
    if not torch.compiler.is_compiling():
        trace = self._reconstruction_stack()
        ids = getattr(trace, '_choice_rule_ids', None)
        valid = getattr(trace, '_choice_mask', None)
        chosen = [TheGrammar.rules[i].canonical for i in sorted(set(ids[valid].tolist()))
                  if i >= 0 and TheGrammar.is_relative_rule(i)] if torch.is_tensor(ids) else []
        calls.append(dict(relative=result.tolist(), selected_relative=chosen,
            rules=self.symbolSpace.current_rules,
            anchored_count=len(getattr(self.wholeSpace, '_anchored_pids', ()) or ())))
    return result
Models.BasicModel._sentence_relative_mask = observed
try:
    TestDepth3RelativeEndState().test_first_trained_read_reaches_depth3_end_state()
finally:
    Path(__file__).with_suffix('.json').write_text(json.dumps(calls, indent=2) + '\n')
