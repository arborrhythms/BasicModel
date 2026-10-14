"""Matched recovery branches from the accepted R/S/T warm-up.

Same seed, revision-3 corpus, optimizer and presentation budgets. The selected-
mint gate defaults on; --gate off is the explicit admission ablation.
Observe closed meaning during the existing held-out pass;
do not repeat warm-up, replay held-out rows, or run identity branches.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import traceback
import warnings

import pytest
import torch
from torch.nn import functional as F

import Models
from identity_corpus import KINDS
from identity_experiment import (
    admission_ablation, component_signal, make_model, present_documents,
    recovery_data, recovery_summary, restore_warmup, save,
)


@contextmanager
def closed_readings(model, docs):
    """Observe the same admitted reading that the identification grader sees."""
    captured = []
    observe = model._sentence_observation

    def observed(state, sid, active, **kwargs):
        result = observe(state, sid, active, **kwargs)
        if kwargs.get('admit'):
            chart = model._concept_owner().components._ready()
            start, end = chart.meaning_start, chart.content_width
            roots = state[1][9][:, sid]
            batch = docs[len(captured):len(captured) + len(result['entries'])]
            for b, doc in enumerate(batch):
                program = result['entries'][b]
                actions = []
                if program is not None:
                    for kind, operation, word in program.actions.detach().cpu().tolist():
                        rules = (model.languageSpace._compose_binary_rules if kind == 1 else
                                 model.languageSpace._compose_unary_rules if kind == 2 else ())
                        actions.append(dict(kind=kind, operation=operation, word=word,
                            method=rules[operation].method_name if rules else None))
                captured.append(dict(id=doc['id'], text=doc['sentences'][int(sid)],
                    closed_meaning_norm=float(roots[b, start:end].detach().norm()),
                    program=actions))
        return result

    model._sentence_observation = observed
    try:
        yield captured
    finally:
        model._sentence_observation = observe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', choices=('factorial', 'confounded'))
    parser.add_argument('out', type=Path)
    parser.add_argument('--gate', choices=('on', 'off'), default='on')
    parser.add_argument('--warmup', type=Path,
        default=Path('output/item6-identity-rst/warmup/warmup.ckpt'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    restore = Models.BaseModel.load_weights
    streams, labels = recovery_data()
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = make_model(args.out, patch)
        try:
            restore_warmup(model, args.warmup, restore)
            signal = component_signal(model)
            if not signal['nonzero_basis_rows'] or not signal['nonzero_observation_rows']:
                raise RuntimeError('warm-up supplied no nonzero component meaning signal')
            component = model._concept_owner().components
            before = set(component.nouns.ids)
            optimizer = model.getOptimizer(lr=.003)
            with admission_ablation(model, off=args.gate == 'off') as admission:
                present_documents(model, streams[args.corpus], optimizer=optimizer, log=args.out)
                minted = set(component.nouns.ids) - before
                model._concept_owner()._online_learning_frozen = True
                with closed_readings(model, streams['heldout']) as readings:
                    rows, _, atoms = present_documents(model, streams['heldout'],
                        recovery_labels=labels, log=args.out)
                report = recovery_summary(rows, component, atoms)
                norms = [r['closed_meaning_norm'] for r in readings]
                programs = Counter(tuple(a['method'] for a in r['program'] if a['method'])
                                   for r in readings)
                names = list(atoms)
                matrix = component.nouns.matrix().detach()
                atom_matrix = torch.stack([atoms[name] for name in names])
                cosines = (F.normalize(matrix, dim=-1) @ F.normalize(atom_matrix, dim=-1).T).abs()
                alignment = [dict(column=cid, minted=cid in minted,
                    abs_cos=dict(zip(names, row.tolist())),
                    nearest_atom=names[int(row.argmax())], nearest_abs_cos=float(row.max()))
                    for cid, row in zip(component.nouns.ids, cosines)]
                report.update(branch=f'recovery-{args.corpus}-{args.gate}', gate=args.gate,
                    warmup=str(args.warmup), warmup_landing='e38a19fbd', seed=20261010,
                    corpus_revision=3, learning_rate=.003,
                    training_documents=len(streams[args.corpus]),
                    heldout_documents=len(streams['heldout']),
                    columns_minted=len(minted), admission=dict(admission),
                    by_kind={name: report['by_atom'][name] for name in KINDS},
                    column_alignment=alignment,
                    atom_norms={name: float(atoms[name].norm()) for name in names},
                    admission_drops=component.admission_drops,
                    closed_meaning=dict(rows=len(norms), nonzero=sum(n > 0 for n in norms),
                        minimum=min(norms), maximum=max(norms), mean=sum(norms)/len(norms)),
                    programs=[dict(methods=list(p), rows=n) for p, n in programs.most_common()],
                    rows=rows, readings=readings, completed=True)
                save(args.out/'results.json', report)
            model.save_weights(str(args.out/'final.ckpt'))
        except Exception:
            save(args.out/'failure.json', dict(error=traceback.format_exc(), completed=False))
            raise
        finally:
            model.End()


if __name__ == '__main__':
    main()
