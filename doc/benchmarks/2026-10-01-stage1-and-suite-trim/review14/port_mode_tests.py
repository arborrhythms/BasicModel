"""Record retired-mode cases and preserve independent live contracts."""
import ast
import json
import re
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
dispositions = []


def retire_file(name, reason):
    p = ROOT / name
    source = p.read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            dispositions.append(dict(file=name, test=node.name, reason=reason,
                old=ast.get_source_segment(source, node)))
    p.unlink()


def retire_functions(name, names, reason):
    p = ROOT / name
    source = p.read_text()
    lines = source.splitlines(keepends=True)
    spans = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            for test in ast.walk(node):
                if isinstance(test, ast.FunctionDef) and test.name.startswith('test_'):
                    dispositions.append(dict(file=name, test=test.name, reason=reason,
                        old=ast.get_source_segment(source, test)))
            spans.append((min([node.lineno]+[x.lineno for x in node.decorator_list])-1, node.end_lineno))
    for start, end in sorted(spans, reverse=True):
        del lines[start:end]
    s = ''.join(lines)
    ast.parse(s)
    p.write_text(s)


old_path = 'Plan §14 retires the old reading mode and its implementation; this case exercises that path only.'
for name in ('test_chunk_static_analyse.py', 'test_analyse_word_learning.py',
             'test_partition_symbolicspace_state.py', 'test_space_equiv_selfcheck.py'):
    retire_file('test/'+name, old_path)
for name in ('test/space_equiv.py', 'test/tools/bpe_gpu_equiv.py', 'test/tools/bpe_gpu_match_check.py'):
    retire_file(name, old_path)
retire_functions('test/test_perceptual_chunking.py', {
    'test_synthesis_mode_lexicon_splits_on_spaces', 'test_synthesis_mode_bpe_returns_learned_segments',
    'test_chunking_invalid_mode_raises', 'test_embedding_token_stream_honors_bpe_fallback_mode'}, old_path)
retire_functions('test/test_perceptualspace_bpe_forward.py', {'TestPerceptualSpaceBPE',
    'test_bpe_constructs_shared_store_with_aligned_ids', 'test_mphf_constructs_shared_store_with_aligned_ids',
    'test_promotion_mirrors_into_store', 'test_bytes_for_round_trips_smoke_prompt_through_store'}, old_path)
# Parameter permanence remains a live native-store invariant; its assertions stay.
p = ROOT/'test/test_perceptualspace_bpe_forward.py'
s = p.read_text().replace('synthesis="bpe"', 'synthesis="meronomy"').replace('self._build_ps("bpe")', 'self._build_ps("meronomy")')
s = s.replace('"""End-to-end BPE tests for PartSpace.', '"""Percept-store permanence and standalone ChunkLayer byte lookup.')
p.write_text(s)
retire_functions('test/test_chunk_layer_bpe.py', {'test_mm_bpe_config_drives_chunk_layer_flags'}, old_path)
retire_functions('test/test_perceptual_loopback.py', {'test_legacy_lexicon_mode_keeps_chunklayer_path'}, old_path)
retire_functions('test/test_input_word_cursor.py', {'test_bpe_word_mask_is_the_validity_signal_when_present'}, old_path)
p = ROOT/'test/test_input_word_cursor.py'; s = p.read_text()
a = s.index('    bpe_mask = (getattr(peer, "_bpe_word_mask", None)')
b = s.index('    any_pos =', a)
s = s[:a]+'    valid_pos = buf.abs().sum(dim=-1) > 0\n'+s[b:]
p.write_text(s)
retire_functions('test/test_analyse_chunking_forward.py', {'test_ws_analysis_knob_accepted'}, old_path)
p = ROOT/'test/test_analyse_chunking_forward.py'; s = p.read_text()
s = s.replace('synthesis="lexicon"', 'synthesis="meronomy"')
s = s.replace('test_lexicon_synthesis_owns_full_surface_lexing', 'test_meronomy_synthesis_owns_full_surface_lexing')
s = s.replace('self._tokens("lexicon", lexer="word")', 'self._tokens("meronomy", lexer="word")')
p.write_text(s)

# Mode-independent tests use the retained production XOR configuration.
# Preserve all assertions; the receipt carries complete old/new test bodies.
old_modes = 'lexicon|radix|bpe|mphf|none|byte|raw|word|sentence|grammatical'
for p in (ROOT/'test').rglob('*.py'):
    if 'test_analyse_chunking_forward.py' == p.name:
        continue
    s = p.read_text()
    new = re.sub(r'<(synthesis|analysis)>('+old_modes+r')</\1>',
                 lambda m:'<'+m[1]+'>meronomy</'+m[1]+'>', s)
    new = new.replace('MM_20M_legacy.xml', 'MM_20M_xor.xml')
    if p.name == 'test_config_matrix.py':
        new = '\n'.join(line for line in new.split('\n') if 'id="legacy"' not in line)
        dispositions.append(dict(file=str(p.relative_to(ROOT)), test='test_config_builds_runs_and_reconstructs[legacy]', reason=old_path))
    if p.name == 'test_use_flags.py':
        new = '\n'.join(line for line in new.split('\n') if '"MM_bpe.xml":' not in line)
        dispositions.append(dict(file=str(p.relative_to(ROOT)), test='test_flags_match_expected [MM_bpe subcase]', reason=old_path))
    # Do not accidentally exclude the retained config because an old exclusion was renamed.
    if p.name == 'test_modality_configs.py':
        new = new.replace('"model.xml", "MM_20M_xor.xml",', '"model.xml",')
    if new != s:
        ast.parse(new)
        p.write_text(new)
(HERE/'mode-test-dispositions.json').write_text(json.dumps(dispositions, indent=2)+'\n')
print('retired cases', len(dispositions))
