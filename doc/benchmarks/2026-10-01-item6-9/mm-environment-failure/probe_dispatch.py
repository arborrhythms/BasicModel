"""Evaluate only the dispatcher's environment assignments, without training."""
import ast
from pathlib import Path

source = Path(__file__).resolve().parents[1] / 'run_measurements.py'
tree = ast.parse(source.read_text())
base = next(n for n in tree.body if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and isinstance(n.value.func.value, ast.Name)
            and n.value.func.value.id == 'env' and n.value.func.attr == 'update')
loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While)
            and any(isinstance(s, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'worker' for t in s.targets)
                    for s in n.body))
assignments = [n for n in loop.body if isinstance(n, ast.Assign)
               and any((isinstance(t, ast.Name) and t.id == 'job_env')
                       or (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                           and t.value.id == 'job_env') for t in n.targets)]
call = next(n for n in ast.walk(loop) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == 'GuardedProcess')
argument = next(n.value for n in call.keywords if n.arg == 'env')
for kind in ('a', 'b', 'c', 'mm'):
    scope = dict(env={}, ROOT=source.parents[3], HERE=source.parent, job=dict(kind=kind))
    exec(compile(ast.Module(body=[base] + assignments, type_ignores=[]), str(source), 'exec'), scope)
    actual = eval(compile(ast.Expression(argument), str(source), 'eval'), scope)['MODEL_COMPILE']
    expected = 'eager' if kind == 'mm' else 'none'
    print(f'{kind}: {actual}; expected {expected}', flush=True)
    assert actual == expected, f'{kind} dispatch inherits incorrect MODEL_COMPILE={actual}'
