"""Counterfactual diagnostics for §12.8, outside the production implementation.

Each fresh process retains the reviewed seed, corpus, and seven-update
protocol. One component changes per process. Overrides and their exact
source are saved beside each measurement; no counterfactual is a gate.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test'), str(HERE)]


def historical_method(path, name):
    source = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT, text=True)
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return textwrap.dedent('\n'.join(source.splitlines()[start - 1:node.end_lineno]) + '\n')


def main(mode, output):
    import Models
    import Interpret
    import ReferenceContext
    import measure
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'identity-no-st':
        module, owner, name = ReferenceContext, ReferenceContext, 'resolve_operand'
        code = textwrap.dedent(inspect.getsource(owner.resolve_operand))
        before = 'resolved = point * (1 + (probability-probability.detach()))[:, :, None]'
        assert before in code
        code = code.replace(before, 'resolved = point')
        scope = 'Remove only the identity softmax straight-through derivative; hard point stays unchanged.'
    elif mode == 'context-head':
        module, owner, name = Models, Models.BasicModel, '_update_contextual_concept_codebooks'
        code = historical_method('bin/Models.py', name)
        scope = 'Use the pre-item-7 context update with no situation/expectation extension; configured weights stay zero.'
    elif mode == 'interpret-head':
        module, owner, name = Interpret, Interpret.InterpretLayer, 'forward'
        code = historical_method('bin/Interpret.py', name)
        code = code.replace('occurrence=None,', 'occurrence=None, selected=None,')
        code = code.replace('    if torch.is_tensor(word):', "    if selected is not None:\n        raise ValueError('counterfactual does not implement selected references')\n    if torch.is_tensor(word):", 1)
        scope = 'Use pre-item-7 interpret admission/order/association. The unused selected keyword is accepted for API shape.'
    elif mode == 'journal-detached':
        module, owner, name = Models, Models.BasicModel, '_tensor_record_operation_values'
        code = textwrap.dedent(inspect.getsource(owner._tensor_record_operation_values))
        before = 'return journal.scatter(1, column, values[:, None])'
        assert before in code
        code = code.replace(before, 'return journal.scatter(1, column, values[:, None]).detach()')
        code = code.replace('@staticmethod\n', '')
        scope = 'Cut only the numerical clause-journal gradient branch; preserve every recorded value and end-state decision.'
    else:
        raise ValueError(mode)
    override = output.with_suffix('.override.py')
    override.write_text(code)
    namespace = dict(vars(module))
    exec(compile(code, str(override), 'exec'), namespace)
    replacement = namespace[name]
    if mode == 'journal-detached':
        replacement = staticmethod(replacement)
    setattr(owner, name, replacement)
    saved = Models.BaseModel.from_config
    configurations = []
    def build(*args, **kwargs):
        model, cfg = saved(*args, **kwargs)
        references = [(r.method_name, list(r.reference_orders)) for r in
            (*model.languageSpace._compose_binary_rules, *model.languageSpace._compose_unary_rules)
            if r.reference_orders]
        settings = dict(reference_rules=references,
            situation_weight=model.contextual_situation_weight,
            expectation_weight=model.contextual_expectation_weight,
            expectation_loss=model.inter_loss_weight,
            sentence_expectation=getattr(model.symbolSpace.discourse, 'expectation_enabled', False))
        configurations.append(settings)
        if mode == 'context-head':
            assert settings['situation_weight'] == settings['expectation_weight'] == 0
        return model, cfg
    Models.BaseModel.from_config = staticmethod(build)
    metadata = dict(mode=mode,scope=scope,override_sha256=hashlib.sha256(code.encode()).hexdigest(),
        base_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    try:
        return measure.baseline(output)
    finally:
        metadata['configurations'] = configurations
        output.with_suffix('.override.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('identity-no-st','context-head','interpret-head','journal-detached'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.mode,args.out))
