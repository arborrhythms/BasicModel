"""Read-only objective measurements, shared by the weekly slow test and receipts.

No RNG draws, optimizer edits, extra updates or configuration edits.
The sole current arm observes §15 ownership. Historical uncut and projected
answer arms are retired; their receipts retain the original measurements.
"""
import argparse
from collections import Counter
import difflib
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import sys
import textwrap
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--config-xml', help='Receipt-local fixture; production XML is never edited')
parser.add_argument('--output', required=True)
parser.add_argument('--arm',choices=('ownership',),default='ownership')
parser.add_argument('--validate-only',action='store_true')
args = parser.parse_args()
OUT = Path(args.output).resolve()
OUT.mkdir(exist_ok=True)
os.environ.pop('BASIC_SEED', None)
os.environ.update(BASICMODEL_DEVICE='cpu', BASIC_AUTOLOAD='false', BASIC_AUTOSAVE='false',
                  MODEL_COMPILE=os.environ.get('MODEL_COMPILE', 'eager'))
import torch
import Models
import Layers
import SentenceCompose
import util
from GradientDiagnostics import _norm, _dot_unit
from bounded_tests import source_snapshot

SOURCE = source_snapshot(ROOT)
(OUT/'source.json').write_text(json.dumps(SOURCE, indent=2))


def serial(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, default=serial))


class Probe:
    def __init__(self):
        self.start = time.monotonic()
        self.training_batches = 0
        self.evaluation_batches = 0
        self.pair_number = 0
        self.captured_first = False
        self.captured_expectation = False
        self.capture_pair = False
        self.pair_rows = []
        self.last_batch = None
        self.last_trial = {}
        self.gradients = []
        self.groups_manifest = {}
        self.eval_reconstructions = []
        self.current = {}
        self.current_bands = []
        self.current_aux = []
        self.truth = {}
        self.model = None
        self.in_batch = False
        self.ownership = {}
        self.ownership_steps = 0
        self.geometry_started = False
        self.geometry_roots = {}
        self.geometry_rows = set()
        self.displacement_steps = 0
        self.stability = {}
        self.epoch = -1

    @torch.no_grad()
    def before_optimizer_step(self, optimizer):
        if self.model is None or not self.current.get('train'):
            return []
        named = dict(self.model.named_parameters())
        codes = {id(cs.similarity_codebook.W) for cs in self.model.conceptualSpaces}
        members = {id(p) for g in optimizer.param_groups for p in g['params']}
        captured = []
        for name, p in named.items():
            if id(p) not in members or (id(p) not in codes and not name.endswith('_anchor') and 'generate_policy' not in name):
                continue
            g = p.grad
            if g is None:
                captured.append((name, p, None, None, None))
            elif g.is_sparse:
                g = g.coalesce()
                assert g.sparse_dim() == 1
                rows = g.indices()[0]
                captured.append((name, p, rows, p.detach().index_select(0, rows).cpu().clone(), g.values().cpu().clone()))
            else:
                captured.append((name, p, None, p.detach().cpu().clone(), g.detach().cpu().clone()))
        return captured

    @torch.no_grad()
    def after_optimizer_step(self, captured):
        if not captured:
            return
        import numpy as np
        folder = OUT/'displacements'
        folder.mkdir(exist_ok=True)
        arrays, rows = {}, []
        for index, (name, p, selected, before, gradient) in enumerate(captured):
            if before is None:
                rows.append(dict(parameter=name, shape=list(p.shape), gradient_present=False,
                                 displacement=0., reason='grad=None: optimizer skips this parameter'))
                continue
            current = p.detach() if selected is None else p.detach().index_select(0, selected)
            delta = current.cpu()-before
            key = f'p{index}'
            arrays[key+'_gradient'] = gradient.float().numpy()
            arrays[key+'_displacement'] = delta.float().numpy()
            if selected is not None:
                arrays[key+'_rows'] = selected.cpu().numpy()
            ng, nd = float(gradient.float().norm()), float(delta.float().norm())
            rows.append(dict(parameter=name, key=key, shape=list(p.shape), gradient_present=True,
                selected_rows=None if selected is None else len(selected),
                gradient_norm=ng, displacement_norm=nd,
                displacement_over_gradient=None if ng==0 else nd/ng,
                cosine=None if ng*nd==0 else float((gradient.float()*delta.float()).sum())/(ng*nd)))
        file = f'step-{self.displacement_steps:05}.npz'
        np.savez_compressed(folder/file, **arrays)
        self.log('optimizer_displacement', step=self.displacement_steps, batch=self.training_batches,
                 file='displacements/'+file, parameters=rows,
                 sparse_rule='unlisted rows have zero gradient and zero displacement; compact row-local optimizer')
        self.displacement_steps += 1

    def derivation(self, model, state, sid, active):
        if not self.in_batch or not model._sentence_training:
            return
        record = model._last_sentence_understanding
        for b in active.nonzero().flatten().tolist():
            # Row and sentence slot identify each configured training stream;
            # word identities distinguish changing native sentences in it.
            word_rows = record.word_rows[b][record.word_valid[b]].detach().cpu().tolist()
            key = json.dumps(word_rows)
            lang = state[1]
            owners, _, _ = model._compose_round_owners(lang[4])
            valid = lang[6][b] & (owners[b] == sid)
            sequence = list(zip(lang[5][b][valid].cpu().tolist(),
                                lang[4][b][valid].cpu().tolist(),
                                lang[17][b][valid].cpu().tolist()))
            signature = json.dumps(sequence)
            row = self.stability.setdefault(key, dict(batch_row=b, sentence_slot=int(sid),
                word_rows=word_rows, epochs={}, derivations=Counter()))
            row['epochs'][self.epoch] = signature
        report=[]
        for row in self.stability.values():
            counts=Counter(row['epochs'].values())
            modal,count=counts.most_common(1)[0]
            report.append(dict(batch_row=row['batch_row'],sentence_slot=row['sentence_slot'],word_rows=row['word_rows'],
                epochs=len(row['epochs']), modal_fraction=count/len(row['epochs']),distinct_derivations=len(counts),
                modal_derivation=json.loads(modal), derivations=[dict(sequence=json.loads(s),epochs=n) for s,n in counts.items()]))
        write('derivation-stability.json',report)

    def capture_ownership(self, model, optimizer):
        report = model.ownership_gradient_diagnostics(optimizer)
        for row in report['parameters']:
            old = self.ownership.setdefault(row['parameter'], dict(row, writers=[]))
            assert old['owner'] == row['owner']
            old['writers'] = sorted(set(old['writers']) | set(row['writers']))
            old['elements'] = row['elements']
            assert not old['writers'] or old['writers'] == [old['owner']]
        self.ownership_steps += 1
        rows = list(self.ownership.values())
        write('ownership.json', dict(parameters=rows, conflicts=0,
            active=sum(bool(row['writers']) for row in rows),
            inactive=sum(not row['writers'] for row in rows),
            backward_steps=self.ownership_steps, scope='all training backwards; endpoint excluded'))

    @torch.no_grad()
    def geometry(self, model, phase):
        """Exact all-row pair moments; full cosine matrices for admitted rows.

        The native reserve has 65,536 rows: materializing its square matrix
        would itself exceed the sweep ceiling. Sum and squared-sum of every
        ordered off-diagonal pair follow from C.T@C, without that allocation.
        """
        books = []
        seen = set()
        for stage, cs in enumerate(model.conceptualSpaces):
            cb = cs.similarity_codebook
            if cb.W.data_ptr() in seen:
                continue
            seen.add(cb.W.data_ptr())
            raw = cb.W.detach().float().cpu()
            norms = raw.norm(dim=-1)
            codes = torch.nn.functional.normalize(raw, dim=-1)
            n = len(codes)
            diagonal = float((norms > 0).sum())
            count = n * (n - 1)
            pair_sum = float(codes.sum(0).double().square().sum()) - diagonal
            gram = codes.T @ codes
            pair_square_sum = float(gram.double().square().sum()) - diagonal
            rows = list(range(n)) if n <= 256 else sorted(r for r in self.geometry_rows if 0 <= r < n)
            selected = codes[rows]
            cosine = selected @ selected.T
            torch.save(dict(rows=rows, pairwise_cosines=cosine, codes=raw[rows]),
                       OUT/f'geometry-{phase}-stage-{stage}.pt')
            vq = getattr(cb, 'vq', None)
            cluster = getattr(vq, 'cluster_size', None)
            cluster_file = None
            if phase == 'end' and torch.is_tensor(cluster):
                cluster_file = f'cluster-size-stage-{stage}.pt'
                torch.save(cluster.detach().cpu(), OUT/cluster_file)
            books.append(dict(stage=stage, rows=n, dimension=raw.shape[-1],
                cosine_matrix_rows=rows, matrix_scope='all rows' if n<=256 else 'all observed primed rows',
                all_dictionary_pairs=dict(count=count, mean=pair_sum/count if count else None,
                    mean_square=pair_square_sum/count if count else None),
                norms=dict(min=float(norms.min()), max=float(norms.max()), mean=float(norms.mean())),
                vq_ema_update=None if vq is None else bool(vq.ema_update),
                cluster_size_file=cluster_file,
                cluster_size=None if cluster is None or cluster.numel()>256 else cluster.detach().cpu().tolist()))
        roots = self.geometry_roots.get(phase)
        write(f'geometry-{phase}.json', dict(dictionary=books,
            roots=None if roots is None else dict(shape=list(roots.shape), values=roots,
                singular_values=torch.linalg.svdvals(roots),
                centered_singular_values=torch.linalg.svdvals(roots-roots.mean(0,keepdim=True))),
            note='No new forward or optimizer step; roots from the measured first trial and endpoint; dictionary pairs are cosine, not dot products.'))

    def log(self, kind, **data):
        with (OUT/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(kind=kind, seconds=time.monotonic()-self.start, **data), default=serial)+'\n')

    def opened(self, model, local):
        self.model = model
        self.in_batch = True
        self.optimizer = local.get('optimizer') or getattr(self, 'optimizer', None)
        self.current = {k:local[k] for k in ('train','split','batchNum','batchSize','outputTensor','what_questions')}
        self.current_bands, self.current_aux, self.truth = [], [], {}
        if self.training_batches==0 and self.evaluation_batches==0:
            self.metadata(model)
            self.groups(model)
        self.log('batch_open', train=local['train'], split=local['split'], batch=local['batchNum'],
                 size=int(local['inputTensor'].shape[0]), supplied=bool(model.inputSpace.data.has_supervised_outputs))

    def groups(self, model):
        named = list(model.named_parameters())
        names = {id(p):name for name,p in named}
        params = {id(p):p for _,p in named}
        def module_parameters(mod):
            return list(mod.parameters()) if isinstance(mod, torch.nn.Module) else []
        codes = []
        percept = module_parameters(model.inputSpace) + module_parameters(model.perceptualSpace)
        vocabulary = module_parameters(getattr(model.outputSpace, '_vocabulary', None))
        output_ids = {id(p) for p in module_parameters(model.outputSpace)}
        # InputSpace registers an OutputSpace back-reference. That does not
        # make the numeric answer head a perception parameter. The shared
        # vocabulary belongs to perception, not the reading map.
        percept = [p for p in percept if id(p) not in output_ids] + vocabulary
        predictors = module_parameters(getattr(model.symbolSpace, 'discourse', None))
        for cs in model.conceptualSpaces:
            codes += module_parameters(getattr(cs, 'similarity_codebook', None))
            percept += module_parameters(getattr(cs, 'concepts_from_percepts', None))
            percept += module_parameters(getattr(cs, 'concept_source_readout', None))
            predictors += module_parameters(getattr(cs, 'intraSentenceLayer', None))
        operation = model.symbolSpace.languageLayer.operation_layer
        ops = list(operation.ops.parameters()) + list(operation.unary_ops.parameters())
        for layer in model.symbolSpace._host_layer_registry.values():
            ops += module_parameters(layer)
        opids, codeids = {id(p) for p in ops}, {id(p) for p in codes}
        chooser = [p for p in operation.parameters() if id(p) not in opids]
        generate = module_parameters(getattr(model.languageSpace, 'generate_policy', None))
        for op in (*model.languageSpace._generate_binary_ops, *model.languageSpace._generate_unary_ops):
            generate += module_parameters(getattr(op, 'gl', op))
        # Groups deliberately overlap when the same physical parameter serves
        # two paths. All intersections are published, never double-counted inside a group.
        vocabulary_ids = {id(p) for p in vocabulary}
        reading = [p for p in module_parameters(model.outputSpace) if id(p) not in vocabulary_ids]
        reading += module_parameters(getattr(model, "answer_record_reader", None))
        if model.answer_synthesis:
            for module in model._question_conditioner_modules():
                reading += module_parameters(module)
        # Include the model's named shared transforms, including perceptual
        # inverses. This read-only ownership view uses no optimizer mutation.
        optimizer = getattr(model, '_sentence_optimizer', None) or self.optimizer
        owners = model.objective_parameter_groups(optimizer)
        shared = {'reconstruction': owners['reconstruction']}
        self.owners = owners
        excluded = codeids | {id(p) for p in predictors + chooser + reading + generate}
        ops += [p for ps in shared.values() for p in ps if id(p) not in excluded]
        groups = dict(perception=[p for p in percept if id(p) not in codeids],
                      codes=codes, chooser=chooser, operators_and_tied_inverses=ops,
                      generate=generate, reading_map=reading,
                      expectation_predictor=predictors)
        used = {id(p) for g in groups.values() for p in g}
        groups['other'] = [p for _,p in named if id(p) not in used]
        groups = {g:list({id(p):p for p in ps}.values()) for g,ps in groups.items()}
        optimizer = getattr(model,'_sentence_optimizer',None)
        owned = {id(p) for pg in optimizer.param_groups for p in pg['params']} if optimizer else set()
        manifest = {g:[dict(name=names.get(id(p),'unregistered'), shape=list(p.shape),
                           requires_grad=p.requires_grad, optimizer_owned=id(p) in owned)
                       for p in ps] for g,ps in groups.items()}
        self.groups_manifest = manifest
        write('parameter-groups.json', manifest)
        write('codebook-ownership.json', [dict(stage=i,
            contextual_rotation_only=bool(getattr(cb,'contextual_rotation_only',False)),
            contextual_learning_rate=float(util.TheXMLConfig.get('architecture.training.conceptualContextLearningRate', default=0) or 0),
            parameters=[dict(name=n,shape=list(p.shape),requires_grad=p.requires_grad)
                        for n,p in cb.named_parameters()],
            buffers=[dict(name=n,shape=list(b.shape),requires_grad=b.requires_grad)
                     for n,b in cb.named_buffers()])
            for i,space in enumerate(model.conceptualSpaces)
            if isinstance((cb:=getattr(space,'similarity_codebook',None)),torch.nn.Module)])
        write('parameter-overlaps.json', {a+'__'+b: [names.get(id(p),'unregistered')
            for p in groups[a] if id(p) in {id(q) for q in groups[b]}]
            for a,b in itertools.combinations(groups,2)})
        return groups, names

    def gradient_report(self, model, objectives, *, scope, trial=None, active=None):
        groups,names = self.groups(model)
        parameters = tuple(dict.fromkeys(p for group in groups.values() for p in group if p.requires_grad))
        versions = [(names.get(id(p),'unregistered'),p._version) for p in parameters]
        digest = hashlib.sha256(json.dumps(versions).encode()).hexdigest()
        rng_state=torch.random.get_rng_state()
        grad_buffers = tuple((id(p.grad), None if p.grad is None else p.grad._version) for p in parameters)
        pullback = getattr(model,'_sentence_pullback',None) if scope=='trial' else None
        gradients, costs, norms = {}, {}, {}
        for objective, cost in objectives.items():
            if torch.is_tensor(cost):
                if cost.ndim:
                    cost = ((cost * active.to(cost)).sum()/active.sum().clamp_min(1)
                            if active is not None else cost.mean())
                costs[objective] = float(cost.detach())
            else:
                costs[objective] = None
            owner = 'output' if objective == 'supplied_answer' else objective
            owned = self.owners.get(owner, ())
            if not torch.is_tensor(cost) or not cost.requires_grad or not owned:
                measured = (None,) * len(owned)
            elif pullback is not None and owner == 'reconstruction':
                measured = pullback.gradients(cost, owned)
            else:
                measured = torch.autograd.grad(cost, owned, retain_graph=True, allow_unused=True)
            by_id = {id(p): g for p, g in zip(owned, measured)}
            gs = [by_id.get(id(p)) for p in parameters]
            gradients[objective] = {p:g.detach() for p,g in zip(parameters,gs) if g is not None}
            norms[objective] = {p:_norm(g) for p,g in gradients[objective].items()}
        report = {}
        per_parameter = {names.get(id(p), 'unregistered'): dict(
            norms={o:norms[o].get(p,0.) for o in objectives},
            unit_dots={a+'__'+b:_dot_unit(gradients[a].get(p),gradients[b].get(p),
                norms[a].get(p,0.),norms[b].get(p,0.))
                for a,b in itertools.combinations(objectives,2)}) for p in parameters}
        for group,ps in groups.items():
            totals = {o:math.hypot(*(norms[o].get(p,0) for p in ps)) for o in objectives}
            cosines = {}
            for a,b in itertools.combinations(objectives,2):
                na,nb=totals[a],totals[b]
                value = None
                if na and nb:
                    value=sum(_dot_unit(gradients[a].get(p),gradients[b].get(p),
                              norms[a].get(p,0),norms[b].get(p,0)) *
                              (norms[a].get(p,0)/na)*(norms[b].get(p,0)/nb) for p in ps)
                    value=max(-1.,min(1.,value))
                cosines[a+'__'+b]=value
            report[group]=dict(norms=totals,cosines=cosines,
                nonzero_parameters={o:[names.get(id(p),'unregistered') for p in ps if norms[o].get(p,0)>0] for o in objectives})
        rng_after=torch.random.get_rng_state()
        torch.random.set_rng_state(rng_state)
        assert versions == [(names.get(id(p),'unregistered'),p._version) for p in parameters]
        assert grad_buffers == tuple((id(p.grad),None if p.grad is None else p.grad._version) for p in parameters)
        entry=dict(scope=scope,trial=trial,pair=self.pair_number,batch=self.training_batches,
                   version_sha256=digest,versions=versions,costs=costs,groups=report,
                   per_parameter=per_parameter,
                   rng_unchanged=bool(torch.equal(rng_state,rng_after)))
        self.gradients.append(entry)
        write('gradients.json',self.gradients)
        self.log('gradient_snapshot', scope=scope,trial=trial,pair=self.pair_number,version_sha256=digest)

    def trial(self, model, local):
        active = local['active']
        registry = model._sentence_cost_registry
        record, recovered = local['record'], local['reconstruction'][0].detach()
        self.geometry_rows.update(int(r) for r in record.primed.rows[record.primed.valid].detach().cpu().tolist())
        root = record.root.detach()[active].float().cpu()
        if not self.geometry_started:
            self.geometry_roots['start'] = root
            self.geometry(model, 'start')
            self.geometry_started = True
        if not model._sentence_training:
            self.geometry_roots['end'] = root
        if model._sentence_reconstruction:
            bank = record.primed
            with torch.no_grad():
                from SentenceUnderstanding import readback_scores
                similarity = torch.stack([readback_scores(recovered[:, w], bank.codes, bank.weights)
                                          for w in range(recovered.shape[1])], 1)
                surface = bank.byte_valid.any(-1) & bank.valid
                own_word = (record.word_rows[..., None] == bank.rows[:, None, :]) & surface[:, None, :]
                activated = surface & ~bank.own
                own_score = similarity.masked_fill(~own_word, -torch.inf).amax(-1)
                extra_score = similarity.masked_fill(~activated[:, None, :], -torch.inf).amax(-1)
                eligible = record.word_valid & active[:, None] & own_word.any(-1)
                competing = eligible & activated.any(-1)[:, None]
                wins = competing & (extra_score > own_score)
                self.log('activated_word_ranking', train=bool(model._sentence_training),
                    trial=model._sentence_trial, pair=self.pair_number,
                    words=int(eligible.sum()), with_activated_candidate=int(competing.sum()),
                    activated_outranks_own=int(wins.sum()),
                    positive_margins=(extra_score-own_score)[wins])
        if model._sentence_reconstruction:
            with torch.no_grad():
                leaves, counts, truncated, _, actions = model._last_decoder_trace
                bank = record.primed
                cosines = torch.nn.functional.cosine_similarity(
                    leaves[:, :, None], bank.codes[:, None], dim=-1)
                lang = local['state'][1]
                owners, _, _ = model._compose_round_owners(lang[4])
                compose_valid = lang[6] & (owners == local['sid'])
                names = (*model.languageSpace._generate_binary_names,
                         *model.languageSpace._generate_unary_names, 'STOP')
                rows = bank.rows.cpu().tolist()
                code_owner = model._concept_owner()
                spellings = [[None if row < 0 else (code_owner.word_surface_for_row(row) or b'').decode('utf8')
                              for row in batch] for batch in rows]
                pairs = {}
                by_name = {name: row for batch, texts in zip(rows, spellings)
                           for row, name in zip(batch, texts) if name}
                for left, right in (('world', 'there'), ('hello', 'loving')):
                    if left in by_name and right in by_name:
                        codes = code_owner.similarity_codebook.lookup_rows(torch.tensor(
                            [by_name[left], by_name[right]], device=leaves.device))
                        pairs[left+'/'+right] = float(torch.nn.functional.cosine_similarity(codes[:1], codes[1:]))
                self.log('decoder', epoch=self.epoch, batch=self.training_batches,
                    trial=model._sentence_trial, train=bool(model._sentence_training), sid=int(local['sid']),
                    rows=rows, surfaces=spellings, priming=bank.weights,
                    pair_cosines=pairs, truncated=truncated,
                    chosen_operations=[[names[i] for i in batch if i >= 0] for batch in actions.cpu().tolist()],
                    compose_rules=[lang[4][b][compose_valid[b]].cpu().tolist() for b in range(len(rows))],
                    compose_arities=[lang[5][b][compose_valid[b]].cpu().tolist() for b in range(len(rows))],
                    leaves=[leaves[b, :int(n)] for b,n in enumerate(counts)],
                    leaf_code_cosines=[cosines[b, :int(n)] for b,n in enumerate(counts)])
        answer = model._sentence_answer_cost
        evaluation_answer = None
        if self.in_batch and not model._sentence_training:
            old = (model._sentence_training, model._sentence_supplied_answers, model._sentence_answer_cost)
            try:
                model._sentence_training = True
                model._sentence_supplied_answers = self.current['outputTensor']
                evaluation_answer = Models.BasicModel._sentence_answer_error(
                    model, local['state'], local['sid'], active, local['observation'])
            finally:
                model._sentence_training, model._sentence_supplied_answers, model._sentence_answer_cost = old
        objectives = dict(reconstruction=registry.total(objective='reconstruction'),
                          expectation=registry.total(objective='expectation'), supplied_answer=answer)
        def rows(value):
            if value is None:
                return [None] * int(active.numel())
            return value.detach().expand(active.shape).cpu().tolist()
        row = dict(pair=self.pair_number, trial=model._sentence_trial,
            train=bool(model._sentence_training), scope='batch' if self.in_batch else 'other_forward',
            batch=self.training_batches, sid=int(local['sid']), active=active.cpu().tolist(),
            weighted={k: rows(v) for k,v in objectives.items()}, terms=registry.breakdown(),
            evaluation_supplied_answer=rows(evaluation_answer),
            reconstruction_enabled=bool(model._sentence_reconstruction))
        self.live_trial = model, objectives, row, active, local['cost']
        if model._sentence_training:
            self.pair_rows.append(row)
            self.last_trial[model._sentence_trial] = row
            if model._sentence_trial == 'exploit':
                e = objectives['expectation']
                live = torch.is_tensor(e) and bool(e.detach().abs().any())
                self.capture_pair = not self.captured_first or (live and not self.captured_expectation)
                self.captured_expectation |= live
                self.captured_first = True

    def after_score(self, result):
        model, objectives, row, active, base = self.live_trial
        self.live_trial = None
        registry = model._sentence_cost_registry
        objectives['expectation'] = registry.total(objective='expectation')
        value = objectives['expectation']
        row['weighted']['expectation'] = ([None]*int(active.numel()) if value is None
                                        else value.detach().expand(active.shape).cpu().tolist())
        row['weighted']['total'] = result[0].detach().cpu().tolist()
        row['terms'] = registry.breakdown()
        self.log('trial', **row)
        if model._sentence_training and self.capture_pair:
            self.gradient_report(model, objectives, scope='trial', trial=model._sentence_trial, active=active)

    def pair(self, saved_costs, wins, active):
        if len(self.pair_rows)!=2:
            return
        a,b=self.pair_rows
        assert a['trial']=='exploit' and b['trial']=='explore'
        if self.capture_pair:
            left,right=self.gradients[-2:]
            assert left['version_sha256']==right['version_sha256'], 'trials not measured at the same parameters'
        self.log('selection',pair=self.pair_number,active=active.cpu().tolist(),wins=wins.cpu().tolist(),
                 costs=saved_costs,exploit=a['weighted'],explore=b['weighted'])
        self.pair_rows=[]
        self.pair_number+=1

    def compute(self, loss, local):
        if self.model is None:
            return
        pred,target=local['pred'].detach(),local['target'].detach()
        nw,ne,nt=int(local['nWhat']),int(local['nWhere']),int(local['nWhen'])
        start=0
        parts={}
        for name,width,weight in [('what',nw,loss.what_scale),('where',ne,loss.where_scale),('when',nt,loss.when_scale)]:
            if width>0:
                raw=float((pred[...,start:start+width]-target[...,start:start+width]).square().mean())
                parts[name]=dict(raw=raw,weight=float(weight),weighted=raw*float(weight))
            start+=width
        caller=inspect.currentframe().f_back.f_back.f_code.co_name
        self.current_bands.append(dict(caller=caller,parts=parts,trial=getattr(self.model,'_sentence_trial',None)))

    def expectation_parts(self, layer, local):
        if not self.in_batch:
            return
        b=local['b']
        kind=None if local['sentence_kinds'] is None else local['sentence_kinds'][b]
        logit=local['kind_logit']
        with torch.no_grad():
            raw=dict(content_mse=float((local['pred'].detach()-local['target'].detach()).square().mean()),
                presence_bce=float(torch.nn.functional.binary_cross_entropy_with_logits(
                    local['logits'].detach(),local['occupied'].to(local['logits']))),
                kind_bce=float(layer._kind_loss(logit.detach() if torch.is_tensor(logit) else logit,
                                               kind,local['cost'].detach())))
        weight=float(self.model.inter_loss_weight)
        self.log('expectation_terms',train=bool(self.model._sentence_training),
            trial=self.model._sentence_trial,pair=self.pair_number,row=b,
            raw=raw,component_weights={key:1. for key in raw},outer_weight=weight,
            weighted={key:value*weight for key,value in raw.items()},
            accounting_residual=float(local['cost'].detach())-sum(raw.values()))

    def auxiliary(self, model, local):
        self.current_aux=[dict(name=n,raw=float(v.detach()),weight=float(w),category=c,
                               enabled=c not in local['pipeline_errors']._disabled)
                          for n,v,w,s,c in local['pipeline_errors'].terms()]

    def truth_return(self, local):
        self.truth={k:serial(v) if torch.is_tensor(v) else v for k,v in local.items()
                    if k in ('multiplier','luminosity_weight','universality_weight','truth_loss_weight',
                             'balance_weight','truth_penalty','balance')}

    def batch(self, model, local):
        terms = model.errors.breakdown()
        total = local['totalLoss']
        raw = dict(totalLoss=float(total.detach()))
        objectives = {name:model.errors.total(objective=name) for name in ('reconstruction','expectation','output')}
        totals = {name:None if value is None else float(value.detach().mean()) for name,value in objectives.items()}
        without_answer = sum(value or 0. for name,value in totals.items() if name != 'output')
        self.log('batch', train=local['train'], split=local['split'], batch=self.training_batches,
            raw=raw, terms=terms, objective_costs=totals, without_answer=without_answer,
            relative_errors=model.errors.total(kind='relative'), penalties=model.errors.total(kind='penalty'))
        self.last_batch = dict(raw=raw, terms=terms, objective_costs=totals,
                              train=local['train'], split=local['split'])
        if local['train']:
            if self.training_batches == 0:
                self.gradient_report(model, dict(reconstruction=objectives['reconstruction'],
                    expectation=objectives['expectation'], supplied_answer=objectives['output']), scope='batch')
            self.training_batches += 1
        else:
            self.evaluation_batches += 1
        self.metadata(model)
        self.in_batch = False

    def metadata(self,model):
        write('weights.json',dict(configuration=args.config, contract='§15: one objective writer per parameter',
            terms=model.errors.breakdown(), trial_terms=getattr(model, '_sentence_cost_registry', Layers.Error()).breakdown(),
            trial_selection='explore R < greedy R; ties keep greedy; reader trains only kept rows',
            normalizers='Error registry: squared error / detached target energy; CE / log K; penalties separate',
            regularizers='Own strengths; concept_readout_l1 is a detached report of the proximal step',
            training_batches=self.training_batches,evaluation_batches=self.evaluation_batches))


P=Probe()
patches=[]
def instrument(cls,name,old,new):
    function=getattr(cls,name)
    before=textwrap.dedent(inspect.getsource(function))
    assert old in before,(name,old)
    after=before.replace(old,new,1)
    namespace=dict(function.__globals__,_objective_probe=P)
    exec(compile(after,str(OUT/(name+'.instrumented.py')),'exec'),namespace)
    setattr(cls,name,namespace[name])
    (OUT/(name+'.instrumented.py')).write_text(after)
    patches.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                    fromfile=cls.__name__+'.'+name+'.original',tofile=cls.__name__+'.'+name+'.observed'))

instrument(Models.BasicModel,'_sentence_path_cost',
    '    return cost, reconstruction, observation, pending',
    '    _objective_probe.trial(self, locals())\n    return cost, reconstruction, observation, pending')
instrument(Models.BasicModel,'_run_batch_once',
    '    self._sentence_answer_questions = what_questions',
    '    self._sentence_answer_questions = what_questions\n    _objective_probe.opened(self, locals())')
# Preserve the already instrumented source for a second insertion without inspect.
function=Models.BasicModel._run_batch_once
# inspect cannot read the unsaved generated path; save the exact full body, then add the next observation.
# The body is recovered from the same original source plus the first insertion.
original = (ROOT/'bin/Models.py').read_text()
import ast
module=ast.parse(original)
klass=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='BasicModel')
node=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='_run_batch_once')
batch_source=textwrap.dedent('\n'.join(original.splitlines()[node.lineno-1:node.end_lineno])+'\n')
batch_source=batch_source.replace('    self._sentence_answer_questions = what_questions','    self._sentence_answer_questions = what_questions\n    _objective_probe.opened(self, locals())',1)
(OUT/'_run_batch_once.instrumented.py').write_text(batch_source)
instrument(Models.BasicModel,'_run_batch_once',
    '        # Snapshot the breakdown before the backward pass',
    '        _objective_probe.batch(self, locals())\n        # Snapshot the breakdown before the backward pass')
if args.config == 'BasicModel_answers_tied_benchmark':
    # Keep the production endpoint pass and its predictions. Its optional
    # console rendering scans the concept bank for every padded position;
    # that rendering is not an objective measurement or an optimizer step.
    instrument(Models.BaseModel, '_reconstructionReport',
        '    if not isinstance(allIn, torch.Tensor) or allIn.numel() == 0:',
        '    if isinstance(allOut, torch.Tensor):\n'
        '        data.reconstructed_output = [x.detach().cpu() for x in allOut]\n'
        '    _objective_probe.log("endpoint_predictions_saved", split=split, rows=n_rows, console_rendering=False)\n'
        '    return\n\n'
        '    if not isinstance(allIn, torch.Tensor) or allIn.numel() == 0:')
(OUT/'instrumentation.patch').write_text(''.join(patches))
original_pair=SentenceCompose.sentence_pair
def paired(cache,compose,score,step,*,active,training=True,before_step=None):
    def observed_score(*args):
        result=score(*args)
        P.after_score(result)
        return result
    value=original_pair(cache,compose,observed_score,step,active=active,training=training,before_step=before_step)
    if training:
        P.pair(value[1],value[2],active)
    return value
SentenceCompose.sentence_pair=paired
original_backward = Models.BaseModel._backward_training_loss
def owned_backward(model, total, amp_scaler=None, *, optimizer=None):
    result = original_backward(model, total, amp_scaler, optimizer=optimizer)
    live = model._sentence_optimizer if getattr(model, '_sentence_backward', False) else optimizer
    P.capture_ownership(model, live)
    return result
Models.BaseModel._backward_training_loss = owned_backward
from Optimizer import MultiOptimizer
original_optimizer_step = MultiOptimizer._step_without_finite_preflight
def observed_optimizer_step(optimizer):
    captured = P.before_optimizer_step(optimizer)
    result = original_optimizer_step(optimizer)
    P.after_optimizer_step(captured)
    return result
MultiOptimizer._step_without_finite_preflight = observed_optimizer_step
original_run_epoch = Models.BasicModel.runEpoch
def observed_run_epoch(model, *args, **kwargs):
    if kwargs.get('optimizer') is not None:
        P.epoch += 1
    return original_run_epoch(model,*args,**kwargs)
Models.BasicModel.runEpoch = observed_run_epoch
original_commit=Models.BasicModel._commit_sentence
def committed(model,state,sid,active,*rest):
    P.derivation(model,state,sid,active)
    if P.in_batch and not model._sentence_training and args.config=='XOR_grammar':
        with torch.no_grad():
            texts,unavailable=model.reconstruct_grammar_sentence(state,sid,active)
        P.eval_reconstructions.append(dict(texts=texts,unavailable=unavailable.cpu().tolist()))
    return original_commit(model,state,sid,active,*rest)
Models.BasicModel._commit_sentence=committed
if args.validate_only:
    write('validated.json',dict(instrumentation_installed=True,model_constructed=False))
    raise SystemExit(0)
config=Path(args.config_xml).resolve() if args.config_xml else ROOT/'data'/(args.config+'.xml')
util.init_config(path=str(config),defaults_path=str(ROOT/'data/model.xml'))
cfg=util.TheXMLConfig.data
arch=cfg.get('architecture',{})
write('plan.json',dict(config=str(config),effective=cfg,seed_applied=None,
    retained_dataset_seed=arch.get('data',{}).get('mathSeed'),
    note='No manual seed. Call existing hydrated factory body after parsing; the XML training seed is not applied.',
    device='cpu',compile=os.environ['MODEL_COMPILE'],runs=1,arm=args.arm,
    comparison='One fresh unseeded model per training arm; not paired initializations; no additional repeats.'))
try:
    results=Models.ModelFactory._run_hydrated(str(config),arch,arch.get('data',{}),arch.get('training',{}))
    model=results[0][2]
    # Native runTrial deliberately omits test passes without BASIC_RUN_TEST.
    # Observe one final pass after training, without changing any optimizer,
    # batch, configuration or training initialization. XOR already evaluates.
    endpoint_added = P.evaluation_batches == 0
    if endpoint_added:
        model.set_sigma(0)
        with torch.no_grad():
            model.runEpoch(batchSize=int(arch['training']['batchSize']), split='test')
        model.set_sigma(0.5)
    P.geometry(model, 'end')
    data=model.inputSpace.data
    P.groups(model)
    write('outcome.json',dict(training_batches=P.training_batches,evaluation_batches=P.evaluation_batches,
        receipt_added_endpoint=endpoint_added,
        last_training_trials=P.last_trial,last_batch=P.last_batch,grammar_reconstructions=P.eval_reconstructions,
        predictions=[serial(x) for x in data.reconstructed_output],targets=[serial(x) for x in data.test_output],
        inputs=[model._bytes_to_text(x).rstrip(chr(0)) for x in data.test_input],seconds=time.monotonic()-P.start))
    assert SOURCE==source_snapshot(ROOT)
    write('complete.json',dict(source_matched=True,seconds=time.monotonic()-P.start))
except BaseException as exc:
    write('error.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.monotonic()-P.start))
    raise
