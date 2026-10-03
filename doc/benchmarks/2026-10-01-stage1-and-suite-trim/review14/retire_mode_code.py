"""Remove the dispatch and implementations retired by plan section 14."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
removed = []


def remove_methods(path, classes):
    p = ROOT / path
    source = p.read_text()
    lines = source.splitlines(keepends=True)
    spans = []
    for cls in ast.parse(source).body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in classes:
            continue
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name in classes[cls.name]:
                start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
                spans.append((start, node.end_lineno))
                removed.append(dict(file=path, name=cls.name+'.'+node.name,
                    old=''.join(lines[start:node.end_lineno]),
                    reason='Old reading mode implementation; no retained dispatch; plan §14.'))
    for start, end in sorted(spans, reverse=True):
        del lines[start:end]
    p.write_text(''.join(lines))


remove_methods('bin/Spaces.py', {
    'Embedding': {'_char_stream'},
    'PartSpace': {'chunk_static', '_analyse_chunk', 'learn_merges', '_embed',
        '_delivered_bytes', '_embed_bpe', '_embed_bpe_gpu', '_embed_bpe_trie',
        '_bpe_emit', '_bpe_finalize', '_bpe_emit_gpu', '_chunk_key_to_latin1',
        '_chunk_to_codebook_idx', '_max_fuse_subtokens', '_mphf_codebook',
        '_mphf_tables', 'mphf_index', 'mphf_table_rows', 'reverse_map_concept',
        '_embed_byte', '_embed_lexicon', '_embed_mphf'},
})
remove_methods('bin/Models.py', {'BaseModel': {'_collect_bpe_extras', '_restore_bpe_extras'},
                                'BasicModel': {'_mphf_route_word'}})

p = ROOT / 'bin/Spaces.py'
s = p.read_text()
s = s.replace("        mode = getattr(self, 'synthesis_mode', 'lexicon')\n        if mode in ('bpe', 'mphf'):\n            return self._char_stream(text)\n", '')

def replace_between(a, b, new=''):
    global s
    assert a in s, a
    start = s.index(a)
    end = s.index(b, start)
    s = s[:start] + new + s[end:]

replace_between('            _bpe_mask_valid = None\n', '            _any_pos =',
                '            _valid_pos = embedded.detach().abs().sum(dim=-1) > 0\n')
replace_between('        # When peer is in BPE chunking mode,', '        # Pad / truncate to N.')
replace_between('        if bpe_mask is not None:\n            valid_mask', '        sub = self.subspace',
                '        valid_mask = (embedded_N.abs().sum(dim=-1) > 0).any(\n            dim=1).reshape(B, 1)\n')
replace_between('        if bpe_mask_N is not None:', '        if len(self._end_of_stream) != B:',
                '        self._word_active_mask = embedded_N.detach().abs().sum(dim=-1) > 0\n')
s = s.replace('        ps._bpe_word_mask_flat = None\n', '')
s = s.replace('        _mode = getattr(ps, "synthesis_mode", None)\n', '')
replace_between('        bpe_mask = (getattr(ps, "_bpe_word_mask", None)', '        with torch.no_grad():')
s = s.replace('            if bpe_mask is not None:\n                _valid_pos = bpe_mask[:, :_Te] > 0\n            elif source_valid_pos is not None:',
              '            if source_valid_pos is not None:')
replace_between('        if bpe_mask is not None:\n            valid_mask', '        was_demuxed =',
'''        if source_valid_pos is not None:
            src_valid = source_valid_pos.to(device=embedded_N.device)
            valid_mask = src_valid[:, :N].any(dim=1).reshape(B, 1)
        else:
            valid_mask = (embedded_N.abs().sum(dim=-1) > 0).any(dim=1).reshape(B, 1)
''')
replace_between('        if bpe_mask_N is not None:', '        self._word_active_mask = _wam',
'''        if source_valid_pos is not None:
            src_valid = source_valid_pos.to(device=embedded_N.device)
            if src_valid.shape[1] >= N:
                _wam = src_valid[:, :N]
            else:
                _wpad = torch.zeros(B, N - src_valid.shape[1],
                                   dtype=torch.bool, device=embedded_N.device)
                _wam = torch.cat([src_valid, _wpad], dim=1)
        else:
            _wam = embedded_N.detach().abs().sum(dim=-1) > 0
''')
s = s.replace('        ps._bpe_word_mask_flat = bpe_mask_N\n', '')
replace_between('        # In BPE mode, re-apply the word-boundary mask', '        # Prime the warm-path cache')
# Remove comments describing deleted implementation blocks.
replace_between('    # Rework A', '    def _slot_forward') if '    # Rework A' in s else None
start = s.index('    def embed_stem', s.index('class PartSpace'))
end = s.index('        self._flush_pending_lexicon_inserts()', start) if '        self._flush_pending_lexicon_inserts()' in s[start:] else -1
ast.parse(s)
p.write_text(s)
p = ROOT / 'bin/Models.py'
s = p.read_text()
s = s.replace('            "bpe_extras": self._collect_bpe_extras(),\n', '')
s = s.replace('        w, _mphf_idx = self._mphf_route_word(w, p)\n        if _mphf_idx is not None:\n            self._mphf_last_idx = _mphf_idx\n            self._mphf_call_count = self._mphf_call_count + 1\n\n', '')
a = s.index('        # MPHF pre-warm (Dynamo-unfriendly build path).')
b = s.index('        # STM bounded-reducer pre-warm:', a)
s = s[:a] + s[b:]
# Both current data checkpoints and both fixture checkpoints have no BPE extras.
for node in ast.walk(ast.parse(s)):
    if isinstance(node, ast.If) and any(isinstance(n, ast.Attribute) and n.attr == '_restore_bpe_extras' for n in ast.walk(node)):
        lines = s.splitlines(keepends=True)
        del lines[node.lineno-1:node.end_lineno]
        s = ''.join(lines)
        break
ast.parse(s)
p.write_text(s)
(HERE/'retired-mode-methods.json').write_text(json.dumps(removed, indent=2)+'\n')
print('removed', len(removed), 'old mode methods')
