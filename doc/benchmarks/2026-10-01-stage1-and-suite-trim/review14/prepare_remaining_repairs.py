"""Prepare section 14.3 source changes, installing only with --apply."""
import ast
import json
import re
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
changes = {}
removed = []


def remove(source, path, names):
    lines = source.splitlines(keepends=True)
    spans = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            start = min([node.lineno]+[n.lineno for n in node.decorator_list])-1
            spans.append((start, node.end_lineno))
            removed.append(dict(file=path, name=node.name,
                old=''.join(lines[start:node.end_lineno]),
                reason='Detached-student / legacy D3 path retired by plan §14.'))
    for a, b in sorted(spans, reverse=True):
        del lines[a:b]
    return ''.join(lines)


def between(source, a, b, replacement):
    assert source.count(a) == 1, (a, source.count(a))
    start = source.index(a)
    end = source.index(b, start)
    return source[:start]+replacement+source[end:]


def save(path, source):
    if path.endswith('.py'):
        ast.parse(source)
    changes[path] = source
    target = HERE/'remaining-preview'/path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)


p = 'bin/Models.py'
s = remove((ROOT/p).read_text(), p,
           {'_detached_reverse_construction_loss', '_d3_reconstruction_loss', '_reverse_from_S', '_left_shift_by_mask'})
s = between(s, '        # Stable construction/reconstruction split.',
    '        self._recon_keep_ideas =', '''        # Every retained serial grammar reading uses its tied inverse.
        self.reconstruction_scope = self._understanding_reconstruction_scope()
        self.reconstruct_in_loop = self.reconstruction_scope == 'understanding'
        self.detached_reverse = False  # diagnostic; no student runtime exists
''')
s = between(s, '        if self.reconstruct_in_loop and self.detached_reverse:',
            '        # The global projection', '')
s = between(s, '        # Declared migration (compiled reverse-loops plan, requirement 5):',
            '        if getattr(self, "reconstruct_in_loop", False):',
'''        # data/BasicModel.ckpt still carries the retired reverse student's
        # twelve parameter keys. Retain this one-way state/optimizer migration;
        # no configuration can instantiate the former training path.
''')
s = s.replace('''        self._sentence_reconstruction = self._sentence_ends and (
            self.reconstruct_in_loop or (self._aligned_serial_word_mode()
                and not self.detached_reverse and self.loss.reconstruction_scale > 0))''',
'''        self._sentence_reconstruction = self._sentence_ends and self.reconstruct_in_loop''')
s = between(s, '            # Rework B (3): on the PER-WORD grammar path',
            '            if self.reconstruct_in_loop:',
'''            # Serial grammar readings consume the owned byte reconstruction.
            # Paths without a grammar keep perceptual reconstruction.
''')
s = s.replace('                self._d3_active = False\n', '')
s = between(s, '            elif d3_loss is not None:',
            '            elif (mask_pos is not None', '')
s = s.replace('f"reconstruction loss zeroed: no D3 (per_word="\n                    f"{bool(_per_word)}) and masked-LM inputs missing "',
              'f"perceptual reconstruction inputs missing "')
s = between(s, '            # Legacy event reconstruction starts',
            '            try:\n                if forwardInput is not None',
'''            # A grammar-free reading retains its perceptual event inverse.
            # The grammar path has already supplied its single byte objective.
            lossRev = torch.zeros((), device=TheDevice.get())
            _rev_dedupe = self.reconstruct_in_loop
''')
s = s.replace('                    and not getattr(self, "detached_reverse", False) \\\n', '')
s = s.replace('''            prediction_errors, contrast_errors = disc._sentence_prediction_errors
            if self.inter_loss_weight > 0:
                errors.merge(prediction_errors, prefix='expectation.', weight=self.inter_loss_weight)
            if self.inter_contrastive_weight > 0:
                errors.merge(contrast_errors, prefix='expectation.', weight=self.inter_contrastive_weight)''',
'''            if self.inter_loss_weight > 0:
                errors.merge(disc._sentence_prediction_errors[0],
                             prefix='expectation.', weight=self.inter_loss_weight)
            if self.inter_contrastive_weight > 0:
                errors.merge(disc._sentence_prediction_errors[1],
                             prefix='expectation.', weight=self.inter_contrastive_weight)''')
save(p, s)
p = 'bin/Language.py'
s = remove((ROOT/p).read_text(), p, {'ReverseConstructionChooser'})
s = between(s, '        # Detached reverse construction student.',
            '        # 7. Sentence expectation',
'''        # Historical checkpoints may contain reverse_chooser keys; loading
        # drops them. The tied inverse has no independent student parameters.
        self.detached_reverse = False
        self.reverse_chooser = None

''')
save(p, s)
p = 'bin/Layers.py'; s = (ROOT/p).read_text()
s = s.replace('''        """Import named constituents, retaining their baselines and ownership."""
        for name, source in other._terms.items():''',
'''        """Import named constituents, retaining their baselines and ownership."""
        if float(weight) == 0.:
            return
        for name, source in other._terms.items():''')
save(p, s)

for p in (ROOT/'data').rglob('*.xml'):
    s = p.read_text()
    new = re.sub(r'^\s*<detachedReverse>[^<]+</detachedReverse>\s*\n', '', s, flags=re.M)
    if new != s:
        save(str(p.relative_to(ROOT)), new)
p = 'data/model.xsd';s = (ROOT/p).read_text()
s = re.sub(r'\s*<xs:element name="detachedReverse"[^>]*/>', '', s)
save(p, s)
(HERE/'remaining-source-deletions.json').write_text(json.dumps(removed, indent=2)+'\n')
(HERE/'remaining-preview-manifest.json').write_text(json.dumps(list(changes), indent=2)+'\n')
if '--apply' in sys.argv:
    for name, source in changes.items():
        (ROOT/name).write_text(source)
print('preview', len(changes), 'files; applied', '--apply' in sys.argv)
