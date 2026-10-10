"""Item 6: fixed forward artifacts, forced inverses and converged free policy.

This is a measurement, not a seed-selected assertion. A failed convergence,
ambiguous inverse, unavailable child, or zero recovery remains in the report.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

PROTOCOL = dict(seed=None, retries=0, chains=[1, 2, 3, 5], operators=['lift', 'lower'],
    max_updates=10000, learning_rate=.01, convergence_ce=.001, convergence_consecutive=5,
    plateau_min_updates=1000, plateau_window=200, plateau_ce_range=1e-6,
    work_limit=32, candidate_limit=16,
    contexts=['stored_dictionary', 'live_forward_wholes'],
    frozen='All roots, words and operator weights; only the generate policy is trained.',
    scoring='Exact ordered codes and multiset codes, completion and continuous leaf error; no pass threshold.')


def tree(op, index, leaves, length):
    result = dict(value=leaves[length-1], code=length-1, children=())
    for row in reversed(range(length-1)):
        left = dict(value=leaves[row], code=row, children=())
        result = dict(value=op.compose(left['value'][None], result['value'][None])[0].detach(),
                      operation=index, children=(left, result))
    return result


def nodes(root):
    pending, result = [root], []
    while pending:
        node = pending.pop()
        result.append(node)
        pending.extend(node['children'])
    return result


@torch.no_grad()
def forced(language, root, words, constituents, *, raw):
    pending = [(root, root['value'][None])]
    values, unavailable, residuals = {}, 0, []
    bank = words[None] if constituents is None else torch.cat((words[None], constituents), 1)
    valid = torch.ones(bank.shape[:2], device=bank.device, dtype=torch.bool)
    while pending:
        node, value = pending.pop()
        if not node['children']:
            values[node['code']] = value[0]
            continue
        index = torch.tensor([node['operation']], device=words.device)
        left, right, bad = language.reverse_binary_step(value, index, torch.tensor([True], device=words.device),
            ops=language._generate_binary_ops, basis=None if raw else bank,
            basis_valid=None if raw else valid, candidate_limit=PROTOCOL['candidate_limit'], return_status=True)
        unavailable += int(bad[0])
        op = language._generate_binary_ops[node['operation']]
        residuals.append(float((op.compose(left, right)-value).square().mean()))
        pending.extend(((node['children'][1], right), (node['children'][0], left)))
    ordered = torch.stack([values[code] for code in sorted(values)])
    distances = (ordered[:, None]-words[None]).square().mean(-1)
    codes = distances.argmin(-1).tolist()
    target = list(range(len(ordered)))
    return dict(exact=codes == target and unavailable == 0, multiset=sorted(codes) == target and unavailable == 0,
        codes=codes, unavailable=unavailable, leaf_mse=float((ordered-words[:len(ordered)]).square().mean()),
        recomposition_mse=max(residuals, default=0.))


@torch.no_grad()
def free(language, root, words, constituents):
    from MemoryIndex import unfold_idea
    result = unfold_idea(language, words, root['value'], PROTOCOL['work_limit'],
        activation=words.new_full((len(words),), 2.), candidate_limit=PROTOCOL['candidate_limit'],
        constituents=constituents,
        constituent_valid=None if constituents is None else torch.ones(constituents.shape[:2], device=words.device, dtype=torch.bool))
    target = list(range(sum(not node['children'] for node in nodes(root))))
    from Generative import reconstruction_coverage
    missing, excess, covered = reconstruction_coverage(torch.tensor(len(result['codes'])),
                                                       torch.tensor(len(target)))
    result['walk_complete'] = result['complete']
    result['missing_constituents'] = int(missing)
    result['excess_constituents'] = int(excess)
    result['coverage_cost'] = float((missing + excess) / max(1, len(target)))
    result['complete'] = result['complete'] and bool(covered)
    result.update(exact=list(result['codes']) == target and result['complete'],
                  multiset=sorted(result['codes']) == target and result['complete'])
    return result


def run(folder):
    from test_prepared_answer_boundary import _native_answer_model
    from test_definition_rows import admit
    from Queries import _basis, _existing_row
    from bounded_tests import source_snapshot
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=PROTOCOL, source=source_snapshot(Path(__file__).resolve().parents[1]), results=[])
    path = folder/'results.json'
    def save():
        path.write_text(json.dumps(report, indent=2)+'\n')
    save()
    model = _native_answer_model(folder, True)
    try:
        language = model.languageSpace
        admitted = [admit(model, name) for name in ('a', 'b', 'c', 'd', 'e')]
        rows = torch.tensor([_existing_row(model._concept_owner(), ('sym', item[1])) for item in admitted])
        words = _basis(model._concept_owner())[rows].detach().clone()
        initial = {name: value.detach().clone() for name, value in language.generate_policy.state_dict().items()}
        artifacts = dict(words=words, initial_policy=initial, roots={}, policies={},
                         initial_model=copy.deepcopy(model.state_dict()), rng=torch.get_rng_state())
        for name in PROTOCOL['operators']:
            index = list(language._generate_binary_names).index(name)
            op = language._generate_binary_ops[index]
            roots = [tree(op, index, words, length) for length in PROTOCOL['chains']]
            artifacts['roots'][name] = roots
            language.generate_policy.load_state_dict(initial)
            stop = len(language._generate_binary_ops)+len(language._generate_unary_ops)
            samples = [node for root in roots for node in nodes(root)]
            inputs = torch.stack([node['value'] for node in samples]).detach()
            targets = torch.tensor([index if node['children'] else stop for node in samples], device=words.device)
            optimizer = torch.optim.Adam(language.generate_policy.parameters(), lr=PROTOCOL['learning_rate'])
            curve, streak, converged, convergence = [], 0, False, 'budget'
            start = time.monotonic()
            for update in range(PROTOCOL['max_updates']+1):
                logits = language.generate_policy_logits(inputs)
                loss = F.cross_entropy(logits, targets)
                accuracy = float(logits.argmax(-1).eq(targets).float().mean())
                curve.append(dict(update=update, ce=float(loss.detach()), accuracy=accuracy))
                streak = streak+1 if float(loss.detach()) <= PROTOCOL['convergence_ce'] and accuracy == 1. else 0
                fitted = streak >= PROTOCOL['convergence_consecutive']
                recent = [item['ce'] for item in curve[-PROTOCOL['plateau_window']:]]
                plateau = (update >= PROTOCOL['plateau_min_updates']
                           and max(recent)-min(recent) <= PROTOCOL['plateau_ce_range'])
                converged = fitted or plateau
                convergence = 'fit' if fitted else 'loss_plateau' if plateau else 'budget'
                if converged or update == PROTOCOL['max_updates']:
                    break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            (folder/(name+'-policy.json')).write_text(json.dumps(curve, indent=2)+'\n')
            artifacts['policies'][name] = copy.deepcopy(language.generate_policy.state_dict())
            conflicting = sum(bool(((inputs == inputs[i]).all(-1) & targets.ne(targets[i])).any())
                              for i in range(len(inputs)))
            for length, root in zip(PROTOCOL['chains'], roots):
                # Forward values are available only to the live-sentence arm.
                # The stored arm gets the dictionary alone, never this list.
                wholes = [node['value'] for node in nodes(root) if node['children'] and node is not root]
                live = None if not wholes else torch.stack(wholes)[None]
                for context in PROTOCOL['contexts']:
                    candidates = live if context == 'live_forward_wholes' else None
                    report['results'].append(dict(operator=name, chain_length=length, depth=length-1,
                        context=context, policy_converged=converged, policy_updates=update,
                        policy_convergence=convergence, conflicting_policy_samples=conflicting,
                        policy_ce=curve[-1]['ce'], policy_accuracy=curve[-1]['accuracy'],
                        raw_forced=forced(language, root, words, None, raw=True),
                        bounded_forced=forced(language, root, words, candidates, raw=False),
                        free=free(language, root, words, candidates)))
            print(name, 'converged', converged, 'updates', update, 'CE', curve[-1]['ce'],
                  'seconds', time.monotonic()-start, flush=True)
            save()
        torch.save(artifacts, folder/'forward-artifacts.pt')
    finally:
        report['source_after'] = source_snapshot(Path(__file__).resolve().parents[1])
        report['source_matched'] = report['source_after'] == report['source']
        save()
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.destination)
