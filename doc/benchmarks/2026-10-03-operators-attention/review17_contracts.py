"""Append-only old/new ports and the explicitly authorized §17 boundaries."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


def seed_calls(text):
    return sorted(ast.unparse(node) for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call) and 'seed' in ast.unparse(node.func).lower())


def methods(text, names):
    return {node.name: ast.dump(node) for node in ast.walk(ast.parse(text))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names}


def main():
    head = git('rev-parse', 'HEAD').strip()
    assert head == 'eb1fbefb5f4a33a22cbb4a590cc0927d6a60761d'
    changed = git('diff', '--name-only', 'HEAD', '--', 'bin', 'test', 'data').splitlines()
    added = git('ls-files', '--others', '--exclude-standard', '--', 'bin', 'test', 'data').splitlines()
    ports, seeds = [], []
    for name in changed + added:
        old = '' if name in added else git('show', 'HEAD:' + name)
        new = (ROOT / name).read_text()
        if name.startswith('test/'):
            ports.append(dict(path=name, old=old, new=new))
        if name.endswith('.py') and seed_calls(old) != seed_calls(new):
            seeds.append(name)
    protected = ['test/test_explicit_dimensions.py', 'test/test_mm_xor.py',
                 'test/test_reconstruction_roundtrip.py', 'test/test_stm_recon_from_cleared_cache.py',
                 'test/pytest_worker.py', 'Makefile', 'pytest.ini',
                 'data/XOR_grammar.xml', 'data/MM_grammar.xml', 'data/model.xml']
    checks = {name: git('show', 'HEAD:' + name) == (ROOT / name).read_text() for name in protected}
    method_checks = {}
    for name, names in [('bin/Models.py', ['_byte_word_cost', '_output_generate_walk']),
                        ('bin/Language.py', ['decoder_eligibility', 'select_logits',
                                             '_distinct_departures'])]:
        old, new = methods(git('show', 'HEAD:' + name), names), methods((ROOT / name).read_text(), names)
        method_checks.update({method: old[method] == new[method] for method in names})
    old_guard = git('show', 'HEAD:test/bounded_tests.py')
    expected = old_guard.replace('report["outcome"] in ("failed", "xpassed")',
                                'report["outcome"] == "failed"').replace(
                                    'r["outcome"] in ("failed", "xpassed")',
                                    'r["outcome"] == "failed"')
    guard_only_xpass = expected == (ROOT / 'test/bounded_tests.py').read_text()
    prior = json.loads((HERE / 'review14-sweep-summary.json').read_text())
    output_files = ('test_prepared_answer_boundary.py', 'test_trial_policy_ownership.py',
                    'test_generation_catalog.py', 'test_output_path_supervised.py',
                    'test_arithmetic_isolation.py')
    output_assertions = {}
    for row in prior['failure_reports']:
        node = row['nodeid']
        name = node.split('::')[0]
        if Path(name).name not in output_files:
            continue
        function = node.split('::')[-1].split('[')[0]
        old = methods(git('show', 'HEAD:' + name), [function])
        new = methods((ROOT / name).read_text(), [function])
        output_assertions[node] = old[function] == new[function]
    assert len(output_assertions) == 16 and all(output_assertions.values())
    result = dict(candidate=head, changed=changed, added=added, complete_test_ports=ports,
                  changed_seed_calls=seeds, protected=checks, unchanged_methods=method_checks,
                  original_output_assertions_unchanged=output_assertions,
                  guard_change_only_non_strict_xpass=guard_only_xpass,
                  resource_limits_unchanged=guard_only_xpass)
    with (HERE / sys.argv[1]).open('x') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    assert not seeds and all(checks.values()) and all(method_checks.values()) and guard_only_xpass
    print(json.dumps({key:value for key,value in result.items() if key != 'complete_test_ports'}))


if __name__ == '__main__':
    main()
