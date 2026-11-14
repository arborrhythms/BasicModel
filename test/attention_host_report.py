"""Render the measured, unchanged-source host inventory with its source context.

Removal labels describe the current representation and a proposed boundary,
not an assertion that an unimplemented tensor rewrite has been certified.
"""
import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path


PHASES = ('preparation', 'walk', 'restoration', 'word step', 'thought', 'generate walks', 'close')
PROTECTED = set('ActionLearning BasicPercepts ChunkJournal ClauseJournal ClauseRow ConceptForgetting '
    'ConceptIndex ConceptLessons ConceptTrust Definitions DerivationReconstruction Examination Extension '
    'FamiliarTier FieldProjection FieldReverse Generative GrammarCheckpoint GrammarConceptCheckpoint '
    'GrammarPhases IdentityRows IndependentComponents Interpret Language Layers MeaningCodes MeaningKernels '
    'Memory MemoryConsolidation MereologicalCodes Mereology Models NumericPercepts Optimizer PartClosing '
    'PartsOfSpeech PerceptCollections PrependedSymbols Queries ReferenceClose ReferenceContext '
    'ReferenceParticipants ScalarPercepts Spaces Taxonomy ThoughtFaces ThoughtStream TimelineRead '
    'WhereRegistry WholePaths data ConceptSpread Expectation Intuition'.split())


def annotation(site):
    module, name = Path(site['file']).stem, site['function']
    text = (name + ' ' + site.get('expression', '') + ' ' + site['source']).lower()
    if module == 'ModelAttention':
        if 'stage_input' in name or 'native_word_poles' in name:
            return ('Python table lookup', 'host stays',
                    'Recover native text positions, dictionary identities and native evidence before the walk.')
        if 'stage_symbol_bank' in name:
            return ('Python table lookup', 'equal rewrite candidate',
                    'Build the symbol bank from native Python IDs; reusing those IDs can avoid reading their tensor copy back.')
    if module == 'Attention' and name in ('finish_narrowing', 'close_narrowing'):
        return ('choice/control', 'equal tensor candidate',
                'Stop on an inert bracket state or resolve field acts in order; device control must preserve that order and RNG.')
    if module in ('GrammarPreference', 'FieldLogic'):
        return ('choice/control', 'equal tensor candidate',
                'Fold bounded structural comparisons or evidence in the existing order; preserve ties and arithmetic order.')
    if module == 'SentenceCompose':
        if 'saved_sentence_values' in name or '_SavedValue' in name:
            return ('Python table lookup', 'host stays',
                    'Deduplicate autograd saves by storage/view/version and protect optimizer-owned storage.')
        return ('Python meanings/references/bindings', 'design change',
                'Traverse heterogeneous sentence carry containers; tensor leaves can be selected after a fixed representation is chosen.')
    if module == 'Meaning' and '_freeze_metadata' in name:
        return ('Python meanings/references/bindings', 'host stays',
                'Own and validate immutable reference/scope metadata, including Python mapping keys and scalar types.')
    if module == 'Meaning':
        return ('Python meanings/references/bindings', 'design change',
                'Construct or validate a Python meaning and its occupied roles/references; a tensor meaning representation would move this boundary.')
    if module == 'ThoughtFeatures' and 'semantic_metadata' in name:
        return ('Python meanings/references/bindings', 'host stays',
                'Alias typed addresses and JSON-encode binding/scope metadata; byte-to-bit tensor construction can be batched exactly.')
    if module in ('SentenceBoundary', 'SentenceDescription', 'MeaningObjects', 'ThoughtReferences') or any(
            token in text for token in ('role_refs', 'bindings', 'constituents', '.roles', 'open_slots')):
        return ('Python meanings/references/bindings', 'design change',
                'Traverse semantic roles, typed references, bindings or nested meaning objects used by the current owner.')
    if module in ('Thoughts', 'ThoughtJournal', 'ThoughtEpisode', 'WalkTrials', 'SentenceFork', 'SentenceCredit') or (
            module.startswith('Thought') and any(token in text for token in ('record', 'steps', 'trace', 'episode'))):
        return ('thought step record', 'design change',
                'Read, restore or append a selected path/episode record; replace the variable Python record with a tensor representation.')
    if module in ('ConceptSpread', 'Queries', 'GrammarMenus') or any(
            token in text for token in ('candidates', 'candidate_', 'menu', 'alternatives', 'rule_order')):
        return ('Python menu', 'design change',
                'Enumerate/filter/order a menu of native rules, candidates or actions before selecting or executing Python code.')
    if module in ('Memory', 'Spaces', 'MereologicalCodes', 'Mereology', 'ConceptIndex', 'Definitions',
                  'WordIdentity', 'WhereRegistry', 'PerceptCollections', 'Interpret', 'ConceptLessons',
                  'FormCentroids', 'ChunkJournal', 'ClauseJournal', 'LexicalRows') or any(
            token in text for token in ('.items()', '.values()', 'lookup', 'dictionary', '_csw_rows', 'definitions')):
        return ('Python table lookup', 'design change',
                'Resolve native identities, rows or definitions through mutable Python tables; a tensor index/registry needs an explicit ownership contract.')
    if module in ('SentenceTraining', 'GradientProjection', 'GradientDiagnostics', 'LearningObjective',
                  'Optimizer', 'Error', 'CompileGuard') or any(token in text for token in ('named_parameters', 'state_dict')):
        return ('choice/control', 'host stays',
                'Manage autograd, optimizer state, validation or named cost registration at the sentence boundary.')
    return ('choice/control', 'design change',
            'Use a selected value or dynamic container to control the current Python owner; moving it requires explicit tensor state/control and an exact replay.')


def enrich(site, root, cache):
    filename = root/'bin'/site['file']
    if filename not in cache:
        source = filename.read_text() if filename.exists() else ''
        cache[filename] = source.splitlines(), [node for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    lines, definitions = cache[filename]
    line = site['line']
    result = dict(site)
    result['source'] = lines[line-1].strip() if 0 < line <= len(lines) else ''
    functions = [node for node in definitions if node.lineno <= line <= node.end_lineno]
    if functions:
        node = min(functions, key=lambda node: node.end_lineno-node.lineno)
        result['function_documentation'] = ast.get_docstring(node) or ''
    if 'expression' in site:
        result['loop_kind'] = site['kind']
        result['operation'] = f"{site['kind']}: {site['expression']}"
    else:
        result['operation'] = f"{site['api']}: {result['source']}"
    result['kind'], result['removal'], result['purpose'] = annotation(result)
    module = Path(site['file']).stem
    result['edit_boundary'] = ('reserved for rounds A/B or plan §5' if module in PROTECTED
                              else 'item 1 only' if module in ('Attention', 'ModelAttention', 'SentenceAttention', 'AttentionTraversal')
                              else 'eligible for an exact local optimization')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measured = json.loads(args.inventory.read_text())
    cache = {}
    hosts = [enrich(site, args.source, cache) for site in measured['host_sites']]
    loops = [enrich(site, args.source, cache) for site in measured['loops']]
    data = dict(source_commit='7a1645e4e6', base_commit='3926005f12',
                sentence='This is Felix.', counts=measured['counts'], host_reads=hosts, loops=loops)
    args.output.with_suffix('.json').write_text(json.dumps(data, indent=2)+'\n')
    out = ['# Native sentence host inventory (2026-11-14)', '',
        'This inventories unchanged production source at `7a1645e4e6`, from `stage_input` through the first kept sentence close/update in the native lesson driver. The fixture, configuration, optimizer and recorded RNG stream are unchanged. The resulting parameter, optimizer and RNG hashes match `3926005f12` exactly.', '',
        'The [complete site ledger](2026-11-14-attention-host-inventory.json) contains every observed host-read API site and Python loop, with its source line, expression, count, purpose, kind, proposed removal class and edit boundary. The table below groups those sites by phase and function. Comprehensions and non-row bookkeeping loops are included as well as row, word and step loops; zero-iteration loop entries are retained.', '',
        'Counts are observations, not estimates. `cpu` on this CPU run may be a no-op; `nonzero` has a dynamic output size. Public tensor API calls, dispatcher operations and host-read API calls are different counting levels and must not be added. The ledger also includes implicit tensor `int`, `float` and `bool` conversions. It does not count internal C++ tensor-library loops as Python loops.', '',
        'Phases use the nearest owning stack frame: attention kernel, restore/snapshot/publication, word step/field read, thought, generate/reconstruction, close/kept output/training finish; the remaining staging and orchestration is preparation. The complete ledger retains the function names and source statements where a different phase split is useful.', '',
        '**Removal classes:** “equal tensor candidate” identifies a bounded numerical/control rewrite that still needs a certificate; “equal rewrite candidate” can reuse existing host input without reading its tensor copy; “design change” means this execution depends on a Python meaning, table, menu, trace or dynamic control representation; “host stays” identifies native text/metadata preparation, validation or autograd/optimizer ownership at the boundary. These conservative labels are not a claim that every candidate has been implemented. The exact byte-feature rewrite retains JSON/address preparation on the host.', '',
        '| Phase | Host API calls | Entered loops/conditions | Yielded iterations/true conditions |',
        '| --- | ---: | ---: | ---: |']
    for phase in PHASES:
        hs = [s for s in hosts if s['phase'] == phase]
        ls = [s for s in loops if s['phase'] == phase]
        out.append(f"| {phase} | {sum(s['count'] for s in hs):,} | {sum(s['entries'] for s in ls):,} | {sum(s['iterations'] for s in ls):,} |")
    for phase in PHASES:
        out += ['', f'## {phase.capitalize()}', '',
                'The loop column is **entries / iterations**; for `while` it is condition evaluations / true conditions. Host API counts are broken down by API. Source-line sets and per-loop counts are in the complete ledger.', '',
                '| Function | Host API counts | Loops | Kind; removal | Purpose |', '| --- | --- | ---: | --- | --- |']
        groups = defaultdict(lambda: dict(host=Counter(), entries=0, iterations=0, sites=[]))
        for site in hosts + loops:
            if site['phase'] != phase:
                continue
            group = groups[site['file'], site['function']]
            if 'api' in site:
                group['host'][site['api']] += site['count']
            else:
                group['entries'] += site['entries']; group['iterations'] += site['iterations']
            group['sites'].append(site)
        for (file, function), group in sorted(groups.items()):
            api = ', '.join(f'{k} {v:,}' for k, v in sorted(group['host'].items())) or '—'
            classifications = sorted({(s['kind'], s['removal']) for s in group['sites']})
            kinds = '; '.join(f'{a}: {b}' for a, b in classifications)
            purposes = ' '.join(dict.fromkeys(s['purpose'] for s in group['sites']))
            out.append(f"| `{file}:{function}` | {api} | {group['entries']:,} / {group['iterations']:,} | {kinds} | {purposes} |")
    args.output.with_suffix('.md').write_text('\n'.join(out)+'\n')
    print(json.dumps(dict(host_sites=len(hosts), loop_sites=len(loops), functions=len({(x['file'], x['function']) for x in hosts+loops}))))


if __name__ == '__main__':
    main()
