"""AST census of the declared attention-to-row evidence path (plan §42).

The inventory is explicit and reviewed. The checker follows local aliases of
the pair and rejects cross-pole arithmetic/comparison, sign, and reductions
over its lane axis. Scalar source authority enters separately and is not a
collapse of identification evidence. Mutation tests exercise the guard.
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIR_FIELDS = {'_attention_poles', '_attention_native_poles', 'poles',
               '_word_reference_evidence', 'leaf_evidence', 'evidence'}
SITES = {
    'bin/ModelAttention.py': {
        'read_poles': ['pair'], 'reference_evidence': ['pair'],
        'handoff': ['native'], 'stage_input': [],
    },
    'bin/Models.py': {
        'BasicModel._attention_sentence_payload': ['pair', 'evidence'],
        'BasicModel._pushed_word_slab': ['pair', 'evidence'],
        'BasicModel._answer_leaf_slab': ['evidence'],
        'BasicModel._capture_reading_programs': [],
        'BasicModel._program_entries': [],
        'BasicModel._publish_sentence_scratch': [],
        'BasicModel._run_sentence_word_bricks': [],
        'BasicModel._run_tensor_peer_word_pipeline': ['leaf_evidence'],
    },
    'bin/Interpret.py': {
        'positive_evidence': [], 'activate_code': ['evidence'],
        'InterpretLayer.activate': ['evidence'],
        'InterpretLayer.forward': ['evidence'], 'InterpretLayer.reverse': ['evidence'],
    },
    'bin/Language.py': {'SymbolSpace.commit_word_reference_slab': ['evidence']},
    'bin/ClauseJournal.py': {'finish_clause': ['pair', 'evidence']},
    'bin/ClauseRow.py': {'ClauseRows.write_clause': ['evidence']},
    'bin/Understanding.py': {'AnswerProgram.__post_init__': []},
    'bin/Layers.py': {'TernaryTruthStore.append_meaning': ['evidence']},
    'bin/MeaningCodes.py': {'exchange': [], 'compose': []},
}


def inspect_site(node, sources):
    both = frozenset(('for', 'against'))
    aliases = {name: both for name in sources}

    def tags(value):
        if isinstance(value, ast.Name):
            return aliases.get(value.id, frozenset())
        if isinstance(value, ast.Attribute):
            if value.attr in PAIR_FIELDS:
                return both
            if value.attr in ('c_plus', 'c_minus'):
                return frozenset(('for' if value.attr == 'c_plus' else 'against',))
            return tags(value.value)
        if isinstance(value, ast.Subscript):
            found = tags(value.value)
            index = value.slice.elts[-1] if isinstance(value.slice, ast.Tuple) else value.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, str) and index.value != 'evidence':
                return frozenset()
            if isinstance(index, ast.Constant) and index.value in (0, 1) and found == both:
                return frozenset(('for' if index.value == 0 else 'against',))
            if isinstance(index, ast.Slice) and isinstance(index.upper, ast.Constant):
                if index.upper.value == 1 and found == both:
                    return frozenset(('for',))
                if index.upper.value == 2 and isinstance(index.lower, ast.Constant) and index.lower.value == 1 and found == both:
                    return frozenset(('against',))
            return found
        # Elementwise validation predicates do not produce evidence magnitudes.
        if isinstance(value, (ast.Compare, ast.BoolOp)):
            return frozenset()
        if isinstance(value, ast.Call):
            name = getattr(value.func, 'attr', getattr(value.func, 'id', ''))
            if name in ('isfinite', 'is_tensor', 'len', 'any', 'all'):
                return frozenset()
            if name in ('read_poles', 'reference_evidence', 'positive_evidence'):
                return both
            if name not in ('detach', 'to', 'clone', 'gather', 'index_select', 'reshape',
                            'reshape_as', 'expand', 'expand_as', 'flip', 'clamp',
                            'clamp_min', 'where', 'stack', 'tensor', 'new_tensor',
                            'tuple', 'map', 'float', 'amin', 'amax', 'minimum', 'maximum',
                            'min', 'max', 'sum', 'mean', 'prod', 'tolist'):
                return frozenset()
        if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) != 2:
            # A payload containing a pair is not itself a pole pair. Its
            # receiving evidence field is an explicit interprocedural seam.
            return frozenset()
        return frozenset().union(*(tags(child) for child in ast.iter_child_nodes(value)))

    def local_nodes(root):
        yield root
        for child in ast.iter_child_nodes(root):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            yield from local_nodes(child)

    # Alias propagation is intentionally local; the explicit inventory defines
    # the interprocedural seams instead of guessing from unrelated numerics.
    for _ in range(12):
        previous = dict(aliases)
        for child in local_nodes(node):
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                value = child.value
                if value is None:
                    continue
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = aliases.get(target.id, frozenset()) | tags(value)
        if aliases == previous:
            break
    failures = []
    def fail(child, reason):
        failures.append(dict(line=child.lineno, reason=reason, expression=ast.unparse(child)))
    for child in local_nodes(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, (ast.Sub, ast.Add, ast.Div, ast.Mult)):
            left, right = tags(child.left), tags(child.right)
            if left and right and left != both and right != both and left != right:
                fail(child, 'arithmetic combines different evidence lanes')
        if isinstance(child, ast.Compare):
            values = [tags(v) for v in (child.left, *child.comparators)]
            if frozenset(('for',)) in values and frozenset(('against',)) in values:
                fail(child, 'comparison collapses different evidence lanes')
        if isinstance(child, ast.Call):
            name = getattr(child.func, 'attr', getattr(child.func, 'id', ''))
            receiver = tags(child.func.value) if isinstance(child.func, ast.Attribute) else frozenset()
            arguments = frozenset().union(*(tags(arg) for arg in child.args))
            if name in ('sign', 'signbit', 'sgn', 'copysign') and (receiver or arguments):
                fail(child, 'takes an evidence sign')
            if name in ('sum', 'mean', 'prod', 'amin', 'amax', 'min', 'max', 'argmin', 'argmax', 'norm', 'vector_norm'):
                method = receiver == both
                pair_argument = bool(child.args) and tags(child.args[0]) == both
                if not method and not pair_argument:
                    continue
                dims = [k.value for k in child.keywords if k.arg in ('dim', 'axis')]
                position = 0 if method else 1
                if not dims and len(child.args) > position:
                    dims = [child.args[position]]
                # The required-evidence read reduces contributions on axis 0,
                # retaining the final pair axis. Explicit per-pole max is safe.
                if not dims or name not in ('amin', 'amax', 'min', 'max') or any(ast.unparse(dim) != '0' for dim in dims):
                    fail(child, 'reduces the evidence lane axis')
    for child in ast.walk(node):
        if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Nested scopes get their own aliases; e.g. metadata's selected
            # evidence must not taint determined_order's selected orders.
            failures.extend(inspect_site(child, sources))
    unique = {(p['line'], p['reason'], p['expression']): p for p in failures}
    return list(unique.values())


def definitions(tree):
    found = {}
    def walk(node, prefix=''):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                found[name] = child
                walk(child, name + '.')
            else:
                walk(child, prefix)
    walk(tree)
    return found


def audit():
    sites, violations = [], []
    for filename, entries in SITES.items():
        source = (ROOT / filename).read_text()
        tree = ast.parse(source)
        found = definitions(tree)
        for name, sources in entries.items():
            node = found[name]
            problems = inspect_site(node, set(sources))
            violations.extend(dict(file=filename, site=name, **p) for p in problems)
            sites.append(dict(file=filename, site=name, line=node.lineno,
                end_line=node.end_lineno, pair_sources=sources,
                sha256=hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest(),
                violations=len(problems)))
    # The retired scalar seam cannot silently reappear under its old name.
    for filename in ('bin/ModelAttention.py', 'bin/Models.py', 'bin/Interpret.py', 'bin/Language.py'):
        for token in ('pole_activation', '_word_reference_activations'):
            if token in (ROOT / filename).read_text():
                violations.append(dict(file=filename, reason='retired scalar seam', expression=token))
    direct_accesses = []
    fields = {'_attention_poles', '_word_reference_evidence', 'leaf_evidence'}
    for path in sorted((ROOT / 'bin').glob('*.py')):
        filename = str(path.relative_to(ROOT))
        for node in ast.walk(ast.parse(path.read_text())):
            field = node.attr if isinstance(node, ast.Attribute) else None
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') in ('getattr', 'setattr') and len(node.args) > 1:
                field = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
            if field not in fields:
                continue
            owners = [site['site'] for site in sites if site['file'] == filename
                      and site['line'] <= node.lineno <= site['end_line']]
            direct_accesses.append(dict(file=filename, line=node.lineno, field=field, audited_by=owners))
            if not owners:
                violations.append(dict(file=filename, line=node.lineno, reason='unlisted evidence consumer', expression=ast.unparse(node)))
    return dict(sites=sites, direct_accesses=direct_accesses, violations=violations,
        rule='no cross-lane arithmetic/comparison, sign, or lane-axis reduction on the declared walk-to-row path',
        boundary='source authority and downstream signed testimony views are separate from identification evidence',
        zero_rules=dict(evidence='min over nonzero contributions per pole', extent_code='coordinatewise min including zero'))
