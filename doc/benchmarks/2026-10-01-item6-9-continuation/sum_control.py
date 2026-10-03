"""One unchanged class-gate run using the declared additive-only control."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(HERE)]
import bounded_tests as bounded
from measure import environment


def child(config):
    from unittest.mock import patch
    import Models
    import test_explicit_dimensions as gates
    original = Models.ModelFactory.run
    with patch.object(Models.ModelFactory, 'run', lambda ignored: original(config)):
        model = gates._run_xor_grammar_in_process()
    data = model.inputSpace.data
    answers = [float(x.reshape(-1)[0]) for x in data.reconstructed_output]
    targets = [float(x.reshape(-1)[0]) for x in data.test_output]
    mse = sum((a-b)**2 for a,b in zip(answers, targets))/4
    correct = sum((a>.5)==(b>.5) for a,b in zip(answers, targets))
    result = dict(answers=answers, targets=targets, mse=mse, correct=correct,
                  class_bar=correct==4 and mse<.05,
                  reconstructions=model._grammar_gate_reconstructions,
                  unavailable=model._grammar_gate_unavailable,
                  rules=model.languageSpace.language_layer.operation_layer.op_names,
                  unary_rules=model.languageSpace.language_layer.operation_layer.unary_names)
    bounded.write_json(HERE / 'sum-control' / 'measurement.json', result)
    # The ordinary class assertion must fail for this negative control.
    assert not result['class_bar'], result


if __name__ == '__main__':
    if len(sys.argv)>1:
        child(sys.argv[1])
    else:
        out = HERE / 'sum-control'
        out.mkdir(exist_ok=False)
        source = bounded.source_snapshot(ROOT)
        old = (ROOT / 'data/XOR_grammar.xml').read_text()
        rules = ('<rule>S = not.forward(S)</rule>\n'
                 '            <rule>S = conjunction.forward(S, S)</rule>\n'
                 '            <rule>S = disjunction.forward(S, S)</rule>')
        assert rules in old
        new = old.replace(rules, '<rule>S = sum.forward(S, S)</rule>')
        config = out / 'XOR_grammar_sum_control.xml'
        config.write_text(new)
        (out / 'control.patch').write_text(''.join(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True),
            fromfile='data/XOR_grammar.xml', tofile='receipt/XOR_grammar_sum_control.xml')))
        bounded.write_json(out / 'source-manifest.json', dict(validated_source=source,
            original_config_sha256=hashlib.sha256(old.encode()).hexdigest(),
            control_config_sha256=hashlib.sha256(new.encode()).hexdigest(),
            epochs=400, seed=None, supplied_answer_training=True,
            interpretation='Sum alone; no unary not, since input-dependent negation is not additive.'))
        worker = bounded.GuardedProcess([sys.executable, str(__file__), str(config)],
            cwd=ROOT, env=environment(ROOT), log_path=out / 'worker.log',
            memory_bytes=8*bounded.GIB, timeout=1800).start()
        result = None
        while result is None:
            result = worker.poll()
            time.sleep(.25)
        assert source == bounded.source_snapshot(ROOT)
        bounded.write_json(out / 'process.json', result)
        print(json.dumps(result))
        raise SystemExit(result['exit_code'])
