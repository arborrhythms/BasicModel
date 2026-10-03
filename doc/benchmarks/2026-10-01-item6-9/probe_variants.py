"""Build reviewable probe patches and apply only their methods in a fresh process.

No production file is written. The saved patched sources make the dynamically
installed method bodies inspectable and reproduce each unified patch exactly.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def methods(source, owner=None):
    tree = ast.parse(source)
    nodes = tree.body if owner is None else next(
        n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    return {n.name: n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def body(source, node):
    return ''.join(source.splitlines(True)[node.lineno - 1:node.end_lineno])


def substitute(source, owner, name, edit):
    node = methods(source, owner)[name]
    lines = source.splitlines(True)
    old = body(source, node)
    new = edit(old)
    assert new != old, name
    return ''.join(lines[:node.lineno - 1]) + new + ''.join(lines[node.end_lineno:])


def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


ANSWER_ERROR = '''
    def _probe_trial_answer_error(self, state, sid):
        """Measurement only: supplied numeric-answer MSE in output units."""
        target = getattr(self, '_probe_supplied_answers', None)
        if (not self._sentence_training or not torch.is_tensor(target)
                or not target.numel() or not self.inputSpace.data.has_supervised_outputs):
            return None
        _stm, lang, _feedback = state
        root = lang[9][:, sid]
        slots = lang[13][:, sid].reshape(root.shape[0], 3, root.shape[-1])
        depth = lang[14][:, sid]
        answer = self._forward_head(None, sentence_state=(root, slots, depth)HEAD_OPTION)
        pred = self.normalizer.denormalize(answer.materialize(), which='output')
        pred = self._align_output_pred(pred, target)
        if pred is None:
            raise RuntimeError('probe answer and supplied target do not align')
        error = (pred - target.to(pred)).square().reshape(root.shape[0], -1).mean(-1)
        self._probe_answer_component = ERROR_VALUE
        return self._probe_answer_component

'''


MULTI_PAIR = '''def sentence_pair(cache, compose, score, step, *, active, training=True):
    """Probe b: greedy plus four independent deviations, all scored before training."""
    exploit = compose(cache, None)
    cost, state = score(exploit, False)
    if cost.shape != active.shape:
        raise ValueError('sentence comparison requires one cost per row')
    costs, states = [cost], [state]
    if training:
        for _ in range(4):
            cost, state = score(compose(cache, exploit), True)
            if cost.shape != active.shape:
                raise ValueError('sentence comparison requires one cost per row')
            costs.append(cost)
            states.append(state)
    saved = torch.stack([cost.detach().clone() for cost in costs], -1)
    # argmin retains the earliest trial on a tie, including greedy at index 0.
    winner = torch.where(active, saved.argmin(-1), torch.zeros_like(active, dtype=torch.long))
    if training:
        for cost in costs:
            step((cost * active.to(cost)).sum() / active.sum().clamp_min(1))
    selected = states[0]
    for index, state in enumerate(states[1:], 1):
        selected = select_rows(selected, state, active & (winner == index))
    return selected, saved, winner
'''


def sources(variant):
    assert variant in ('a', 'b', 'c')
    original = {name: (ROOT / 'bin' / name).read_text()
                for name in ('Models.py', 'SentenceCompose.py')}
    changed = dict(original)
    model = substitute(changed['Models.py'], 'BasicModel', '_run_batch_once', lambda s:
        once(s, '        inputTensor, outputTensor = batch\n',
             '        inputTensor, outputTensor = batch\n'
             '        self._probe_supplied_answers = outputTensor\n'))
    helper = ANSWER_ERROR.replace('HEAD_OPTION', ', detach_understanding=False' if variant == 'c' else '')
    helper = helper.replace('ERROR_VALUE', 'error' if variant == 'c' else 'error.detach()')
    model = substitute(model, 'BasicModel', '_sentence_path_cost', lambda s:
        once(s, '        return cost, reconstruction, observation, pending\n',
             '        answer_error = self._probe_trial_answer_error(state, sid)\n'
             '        if answer_error is not None:\n'
             '            cost = cost + answer_error\n'
             '        return cost, reconstruction, observation, pending\n') + helper)
    if variant == 'c':
        model = substitute(model, 'BasicModel', '_forward_head', lambda s:
            once(once(s, 'sentence_state=None):',
                      'sentence_state=None, detach_understanding=True):'),
                 'value = slots.detach().reshape(B, 3 * D)',
                 'value = (slots.detach() if detach_understanding else slots).reshape(B, 3 * D)'))
    if variant == 'b':
        def commit(s):
            s = once(s, 'observations[-1 if win else 0]', 'observations[int(win)]')
            s = once(s,
                "            pending_a, pending_b = predictions[0], predictions[-1]\n"
                "            disc._inter_last_meaning = [pending_b[0][b] if win else pending_a[0][b]\n"
                "                                       for b, win in enumerate(rows)]\n"
                "            disc._inter_last_pred_root = [pending_b[1][b] if win else pending_a[1][b]\n"
                "                                         for b, win in enumerate(rows)]\n",
                "            disc._inter_last_meaning = [predictions[int(win)][0][b]\n"
                "                                       for b, win in enumerate(rows)]\n"
                "            disc._inter_last_pred_root = [predictions[int(win)][1][b]\n"
                "                                         for b, win in enumerate(rows)]\n")
            return s
        model = substitute(model, 'BasicModel', '_commit_sentence', commit)
        changed['SentenceCompose.py'] = substitute(changed['SentenceCompose.py'], None,
                                                  'sentence_pair', lambda s: MULTI_PAIR)
    changed['Models.py'] = model
    return original, changed


def save():
    out = HERE / 'probe-patches'
    out.mkdir(exist_ok=False)
    for variant in ('a', 'b', 'c'):
        original, changed = sources(variant)
        patch, ledger = [], []
        for name, value in changed.items():
            ast.parse(value)
            path = out / variant / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(value)
            patch.extend(difflib.unified_diff(original[name].splitlines(True), value.splitlines(True),
                fromfile='a/bin/' + name, tofile='b/bin/' + name))
            owner = 'BasicModel' if name == 'Models.py' else None
            old_nodes, new_nodes = methods(original[name], owner), methods(value, owner)
            for symbol, node in new_nodes.items():
                old = body(original[name], old_nodes[symbol]) if symbol in old_nodes else None
                new = body(value, node)
                if old != new:
                    ledger.append(dict(file='bin/' + name, symbol=symbol, old_body=old, new_body=new))
        (out / f'{variant}.patch').write_text(''.join(patch))
        (out / f'{variant}-bodies.json').write_text(json.dumps(ledger, indent=2) + '\n')
    (out / 'source-hashes.json').write_text(json.dumps({name: hashlib.sha256(
        (ROOT / 'bin' / name).read_bytes()).hexdigest() for name in ('Models.py', 'SentenceCompose.py')}, indent=2) + '\n')


def install(variant, model_module, compose_module):
    manifest = json.loads((HERE / 'probe-patches/source-hashes.json').read_text())
    for name, module, owner in (('Models.py', model_module, 'BasicModel'),
                                 ('SentenceCompose.py', compose_module, None)):
        assert hashlib.sha256((ROOT / 'bin' / name).read_bytes()).hexdigest() == manifest[name]
        path = HERE / 'probe-patches' / variant / name
        source = path.read_text()
        original = (ROOT / 'bin' / name).read_text()
        old_nodes = methods(original, owner)
        for symbol, node in methods(source, owner).items():
            if symbol in old_nodes and body(source, node) == body(original, old_nodes[symbol]):
                continue
            namespace = vars(module)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
            setattr(getattr(module, owner) if owner else module, symbol, namespace[symbol])


if __name__ == '__main__':
    save()
