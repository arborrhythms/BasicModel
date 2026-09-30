"""One final source-matched full sweep after measurements and explicit gates."""
import importlib.util
import json
from pathlib import Path
import sys

from review_source import supporting_inputs

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('review_runner', HERE / 'run_rename_sweep.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    gate_dir = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'explicit')
    output = HERE / (sys.argv[2] if len(sys.argv) > 2 else 'final')
    gates = json.loads((gate_dir / 'result.json').read_text())
    assert gates['reason'] != 'running' and not gates.get('active_workers')
    manifest = json.loads((gate_dir / 'source-manifest.json').read_text())
    assert manifest['validated_source'] == runner.bounded.source_snapshot(runner.ROOT)
    inputs = supporting_inputs(runner.ROOT)
    assert inputs == manifest['supporting_inputs']
    runner.HERE = output
    runner.HERE.mkdir(exist_ok=False)
    (output / 'supporting-inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
    result = runner.main()
    assert supporting_inputs(runner.ROOT) == inputs, 'fixture inputs changed during sweep'
    assert manifest['validated_source'] == runner.bounded.source_snapshot(runner.ROOT)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
