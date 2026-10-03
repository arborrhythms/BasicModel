"""Probe (repository untouched). Train XOR_grammar as the gate does, logging every operation choice of both
trials; rebuild each derivation from the logged stack; enumerate every full derivation over the same leaves;
report which derivations make XOR readable with a margin, and whether training ever uses them."""
import os, sys, warnings, json, io, contextlib, re, itertools, collections
tree = sys.argv[1]; config = sys.argv[2] if len(sys.argv) > 2 else "data/XOR_grammar.xml"
os.chdir(tree); sys.path.insert(0, tree + '/bin'); sys.path.insert(0, tree + '/test')
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("BASICMODEL_DEVICE", "cpu"); os.environ["MODEL_COMPILE"] = "none"
import torch; warnings.filterwarnings("ignore")
import Models, Language, Spaces

TARGET = ['hello world', 'hello there', 'loving world', 'loving there']
Y = torch.tensor([-1., 1., 1., -1.], dtype=torch.float64)   # head space: answer = (h + 1) / 2
REACH = float(os.environ.get('REACH', '5'))                  # weight size the answer map reaches (measured .4-4.9)
STATE = dict(model=None, items=None, batch=-1)
CALLS = []

orig_prep = Spaces.InputSpace.prepInput
def prep(self, items, *a, **k):
    try: STATE['items'] = [str(x).replace('\x00', '').strip() for x in items]
    except Exception: STATE['items'] = None
    return orig_prep(self, items, *a, **k)
Spaces.InputSpace.prepInput = prep
orig_rb = Models.BasicModel.runBatch
def rb(self, *a, **k):
    STATE['model'] = self; STATE['batch'] += 1
    return orig_rb(self, *a, **k)
Models.BasicModel.runBatch = rb
orig_sel = Language.OperationSelectionLayer.forward
def sel(self, x, **kw):
    hard, path, routing = orig_sel(self, x, **kw)
    m = STATE['model']
    CALLS.append(dict(batch=STATE['batch'], training=None if m is None else bool(m.training),
        trial=None if m is None else getattr(m, '_sentence_trial', None), items=STATE['items'],
        x=x.detach().double().clone(), depth=None if kw.get('depth') is None else kw['depth'].detach().reshape(-1).tolist(),
        kind=routing['kind'].reshape(-1).tolist(), op=routing['op'].reshape(-1).tolist(), pos=routing['position'].reshape(-1).tolist(),
        valid=routing['valid'].reshape(-1).tolist() if 'valid' in routing else None,
        names=(list(self.op_names or []), list(self.unary_names or []))))
    return hard, path, routing
Language.OperationSelectionLayer.forward = sel

def margin(U):
    """Smallest weights an affine head needs to give XOR exactly from the four roots (None if impossible)."""
    X = torch.cat((U, torch.ones(U.shape[0], 1, dtype=torch.float64)), 1)
    w = torch.linalg.pinv(X, rtol=1e-10) @ Y
    err = float(((X @ w - Y) ** 2).mean())
    sv = torch.linalg.svdvals(U - U.mean(0, keepdim=True))
    return dict(exact=err < 1e-6, norm=float(w[:-1].norm()), max=float(w[:-1].abs().max()),
                sv3=float(sv[2]) if len(sv) > 2 else 0.0)

BIN = {'conjunction': torch.minimum, 'disjunction': torch.maximum, 'intersection': torch.minimum, 'union': torch.maximum}
def simulate(segment):
    """Rebuild each row's stack through one trial's calls. Returns per row (stack, leaves, max deviation)."""
    B = len(segment[0]['kind'])
    stacks = [[] for _ in range(B)]; leaves = [[] for _ in range(B)]; dev = [0.0] * B
    for c in segment:
        binary, unary = c['names']
        for b in range(B):
            d = c['depth'][b] if c['depth'] is not None else None
            if d is None: continue
            st = stacks[b]
            # the slab shows the stack as the chooser sees it; check what we rebuilt, then push new leaves
            for i, (e, v) in enumerate(st[:d]):
                dev[b] = max(dev[b], float((c['x'][b, i, :v.shape[0]] - v).abs().max()))
            while len(st) < d:
                v = c['x'][b, len(st)].clone(); name = f"L{len(leaves[b])}"
                leaves[b].append(v); st.append((name, v))
            if c['valid'] is not None and not c['valid'][b]: continue
            k, o, p = c['kind'][b], c['op'][b], c['pos'][b]
            if k == 1 and p + 1 < len(st):
                name = binary[o] if o < len(binary) else f'bin{o}'
                f = BIN.get(name)
                if f is None: st[p:p + 2] = [(f"{name}({st[p][0]},{st[p+1][0]})", st[p][1])]; continue
                st[p:p + 2] = [(f"{'min' if f is torch.minimum else 'max'}({st[p][0]},{st[p+1][0]})", f(st[p][1], st[p + 1][1]))]
            elif k == 2 and p < len(st):
                name = unary[o] if o < len(unary) else f'un{o}'
                if name == 'not': st[p] = (st[p][0][4:-1] if st[p][0].startswith('not(') else f"not({st[p][0]})", -st[p][1])
                else: st[p] = (f"{name}({st[p][0]})", st[p][1])
    return stacks, leaves, dev

def enumerate_full(leafvals):
    """Every full reduction of the ordered leaves with min/max nodes and negations; -> {expr: [4 roots]}."""
    n = len(leafvals[0]); out = {}
    def trees(lo, hi):
        if hi - lo == 1:
            yield (f"L{lo}", lambda L, i=lo: L[i]); return
        for mid in range(lo + 1, hi):
            for (le, lf), (re_, rf) in itertools.product(list(trees(lo, mid)), list(trees(mid, hi))):
                for opn, f in (('min', torch.minimum), ('max', torch.maximum)):
                    for nl, nr in itertools.product((0, 1), repeat=2):
                        a = f"not({le})" if nl else le; b = f"not({re_})" if nr else re_
                        yield (f"{opn}({a},{b})", (lambda L, lf=lf, rf=rf, f=f, nl=nl, nr=nr:
                               f(-lf(L) if nl else lf(L), -rf(L) if nr else rf(L))))
    for e, fn in trees(0, n):
        out[e] = torch.stack([fn(L) for L in leafvals])
    return out

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    results = Models.ModelFactory.run(config)
m = results[0][2]
pairs = [(float(a), float(b)) for a, b in re.findall(r"label=([0-9.]+) predicted=([-0-9.]+)", buf.getvalue())][-4:]
answers = [b for _, b in pairs]; mse = sum((a - b) ** 2 for a, b in pairs) / 4

# group calls into trials: consecutive calls with the same (batch, training, trial)
segments = []
for c in CALLS:
    key = (c['batch'], c['training'], c['trial'])
    if segments and segments[-1][0] == key: segments[-1][1].append(c)
    else: segments.append((key, [c]))

def align(items):
    return [items.index(t) for t in TARGET] if items and all(t in items for t in TARGET) else None

train_rows = []   # per training trial: expressions per sentence, margin of its roots, best margin available
best_cache = {}
for (batch, training, trial), seg in segments:
    if not training or trial not in ('exploit', 'explore'): continue
    order = align(seg[0]['items'])
    if order is None: continue
    stacks, leaves, dev = simulate(seg)
    if any(len(stacks[i]) != 1 for i in order):
        train_rows.append(dict(batch=batch, trial=trial, full=False, exprs=[tuple(e for e, _ in stacks[i]) for i in order])); continue
    exprs = [stacks[i][0][0] for i in order]
    U = torch.stack([stacks[i][0][1] for i in order])
    mg = margin(U)
    if batch not in best_cache and trial == 'exploit' and all(len(leaves[i]) == len(leaves[order[0]]) for i in order):
        enum = enumerate_full([leaves[i] for i in order])
        best_cache[batch] = min((margin(R)['max'] if margin(R)['exact'] else float('inf'), e) for e, R in enum.items())
    train_rows.append(dict(batch=batch, trial=trial, full=True, exprs=exprs, uniform=len(set(exprs)) == 1,
                           margin_max=mg['max'] if mg['exact'] else float('inf'), sv3=mg['sv3'], dev=max(dev[i] for i in order)))

# the final evaluation (the report's last eval pass): greedy derivations and leaves
evals = [(k, s) for k, s in segments if k[1] is False]
summary = dict(answers=[round(a, 3) for a in answers], mse=round(mse, 4), trials_logged=len(train_rows))
if evals:
    (_, _, _), seg = evals[-1]
    order = align(seg[0]['items'])
    stacks, leaves, dev = simulate(seg)
    if order is not None:
        exprs = [tuple(e for e, _ in stacks[i]) for i in order]
        summary['greedy_derivations'] = exprs
        summary['rebuild_max_deviation'] = round(max(dev[i] for i in order), 6)
        summary['leaf_count'] = [len(leaves[i]) for i in order]
        summary['leaves'] = [[[round(float(v), 3) for v in leaf] for leaf in leaves[i]] for i in order]
        if all(len(stacks[i]) == 1 for i in order):
            mg = margin(torch.stack([stacks[i][0][1] for i in order]))
            summary['greedy_margin'] = dict(exact=mg['exact'], max_weight=round(mg['max'], 2), sv3=round(mg['sv3'], 4))
            S = getattr(m, '_stm_single_S', None)
            if torch.is_tensor(S) and S.shape[0] == 4:
                summary['root_matches_model'] = round(float((torch.stack([stacks[i][0][1] for i in order]) - S.detach().double()[order][:, :stacks[order[0]][0][1].shape[0]]).abs().max()), 6)
        enum = enumerate_full([leaves[i] for i in order])
        ranked = sorted(((margin(R), e) for e, R in enum.items()), key=lambda t: (not t[0]['exact'], t[0]['max']))
        summary['expressions_enumerated'] = len(ranked)
        summary['readable_within_reach'] = sum(1 for g, e in ranked if g['exact'] and g['max'] <= REACH)
        summary['best'] = [(e, round(g['max'], 2), round(g['sv3'], 3)) for g, e in ranked[:8]]
        summary['worst_exact'] = [(e, round(g['max'], 1)) for g, e in ranked if g['exact']][-3:]
        summary['not_readable'] = sum(1 for g, e in ranked if not g['exact'])
        if 'greedy_derivations' in summary and all(len(x) == 1 for x in exprs) and len(set(exprs)) == 1:
            names = [e for g, e in ranked]
            summary['greedy_rank'] = names.index(exprs[0][0]) + 1 if exprs[0][0] in names else None
# training: did any trial use a derivation the answer could read within reach?
full = [r for r in train_rows if r['full']]
for trial in ('exploit', 'explore'):
    rows = [r for r in full if r['trial'] == trial]
    if not rows: continue
    within = [r for r in rows if r['margin_max'] <= REACH]
    summary[f'{trial}_trials'] = len(rows)
    summary[f'{trial}_readable_within_reach'] = len(within)
    summary[f'{trial}_distinct_derivations'] = len({tuple(r['exprs']) for r in rows})
    summary[f'{trial}_top'] = collections.Counter(r['exprs'][0] for r in rows if r['uniform']).most_common(4)
    summary[f'{trial}_rebuild_max_deviation'] = round(max(r['dev'] for r in rows), 6)
avail = [v[0] for v in best_cache.values()]
summary['best_available_first_last'] = [best_cache[min(best_cache)], best_cache[max(best_cache)]] if best_cache else None
summary['epochs_where_some_derivation_is_readable_within_reach'] = f"{sum(a <= REACH for a in avail)}/{len(avail)}"
summary['best_available_max_weight_median'] = sorted(avail)[len(avail) // 2] if avail else None
summary['best_available_examples'] = [best_cache[k] for k in sorted(best_cache)[::100]]
summary['partial_trials'] = sum(1 for r in train_rows if not r['full'])
print("DERIV " + json.dumps(summary, default=str))
