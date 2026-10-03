"""Complete unit-based meronomy for the mixing serial binding (receipt probes saved)."""
from pathlib import Path
import ast,sys
R=Path(__file__).resolve().parent;ROOT=R.parents[3]
changes={}
def save(p,s):
 ast.parse(s);changes[p]=s
 target=R/'migration-preview'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
p='bin/Models.py';s=(ROOT/p).read_text()
a='''        in_sub, concepts_in = self.inputSpace.forward(x)'''
b='''        # Both bindings traverse analysis units. The mixing binding uses
        # its already configured PS word slots; native binding keeps its
        # explicit serialWordCapacity/buckets. Identity publication is still
        # controlled solely by the binding, not by this input representation.
        ps = self.perceptualSpace
        object.__setattr__(ps, '_serial_reading', bool(self.serial))
        if self.serial and not self.serial_object_meta:
            object.__setattr__(ps, '_serial_word_capacity', int(ps.outputShape[0]))
        in_sub, concepts_in = self.inputSpace.forward(x)'''
assert s.count(a)==1;s=s.replace(a,b)
save(p,s)
p='bin/Spaces.py';s=(ROOT/p).read_text()
s=s.replace('''        if (getattr(self, "_serial_object_meta", False)
                and int(getattr(self, "_serial_word_capacity", 0) or 0) > 0):
            return self._embed_ladder_word_major(upstream_vspace)''','''        if ((getattr(self, "_serial_object_meta", False)
                 or getattr(self, "_serial_reading", False))
                and int(getattr(self, "_serial_word_capacity", 0) or 0) > 0):
            return self._embed_ladder_word_major(upstream_vspace)''')
a='''            word_texts_rows.append([raw[s0:e0].decode("latin1") for s0, e0 in spans])'''
b='''            if not getattr(self, '_serial_object_meta', False):
                # The analysis may omit separator runs from its property
                # wholes. Perception still owns those bytes. Preserve the
                # chosen cuts and insert each uncovered span as one unit;
                # the mixing grammar separately excludes whitespace leaves.
                complete, cursor = [], 0
                for start, end in sorted(spans):
                    if start > cursor:
                        complete.append((cursor, start))
                    complete.append((start, end))
                    cursor = end
                if cursor < len(raw):
                    complete.append((cursor, len(raw)))
                spans = complete
            word_texts_rows.append([raw[s0:e0].decode("latin1") for s0, e0 in spans])'''
assert s.count(a)==1;s=s.replace(a,b)
s=s.replace('''        # bytes; no trie lookup, no promotion.  The non-word-major path is
        # still radix-backed (pending).''','''        # bytes. Both serial bindings use that unit-based stem; a whole-slab
        # reading keeps its native percept-store emission.''')
save(p,s)
p='bin/WhereRegistry.py';s=(ROOT/p).read_text()
a="""    registry = WhereRegistry((
        ('input', int(model.inputSpace.outputShape[0])),"""
b="""    input_extent = int(model.inputSpace.outputShape[0])
    if model.serial and not bool(TheXMLConfig.get('architecture.serialObjectMeta', default=False)):
        # A mixing serial InputSpace declares word slots, whereas occurrence
        # coordinates are byte starts. Convert the existing unit/atom bounds
        # to byte addresses; do not compare a byte offset with a word count.
        # The word and per-word residual limits remain independently enforced.
        word_capacity = int(model.perceptualSpace.outputShape[0])
        part_capacity = int(TheXMLConfig.get('architecture.serialResidualPartCapacity', default=16))
        input_extent = max(input_extent, word_capacity * part_capacity)
    registry = WhereRegistry((
        ('input', input_extent),"""
assert s.count(a)==1;s=s.replace(a,b)
save(p,s)
p='test/test_generation_lesson.py';s=(ROOT/p).read_text().replace('torch.autograd.grad(output, generate, retain_graph=True)', 'torch.autograd.grad(output.mean(), generate, retain_graph=True)');save(p,s)
p='test/test_sentence_compose.py';s=(ROOT/p).read_text()
a='''        fixed = cost.new_tensor([1., 2.] if alternative else [2., 1.])
        return cost - cost.detach() + fixed, recon, observation, pending'''
b='''        fixed = cost.new_tensor([1., 2.] if alternative else [2., 1.])
        # This causal fixture fixes the comparison, not the numerical reading.
        # Give the two trials equal reconstruction so the decided precedence
        # rule permits the same controlled winners; keep its live derivative.
        recon = (*recon[:2], recon[2] - recon[2].detach() + 1., *recon[3:])
        return cost - cost.detach() + fixed, recon, observation, pending'''
assert s.count(a)==1;s=s.replace(a,b);save(p,s)
p='test/test_grammar_separator.py';s=(ROOT/p).read_text()
# The read-back-only inverse has no scoring targets. Observe the actual
# reconstruction objective before that later, targetless decode is called.
s=s.replace('''        targets.append(target_bytes.detach().clone())''','''        if ready:
            targets.append(target_bytes.detach().clone())''')
s=s.replace('''        assert isp._word_active_mask[:, :3].tolist() == [[True, True, True]] * 4''','''        assert isp._word_active_mask[:, :3].tolist() == [[True, True, True]] * 4
        assert isp._word_active_mask.sum(-1).tolist() == [4 if trailing else 3] * 4
        assert model.where_registry.slices['input'][1] == (
            int(model.perceptualSpace.outputShape[0]) * model.serial_residual_part_capacity)''')
save(p,s)
if '--apply' in sys.argv:
 for p,s in changes.items():(ROOT/p).write_text(s)
print('prepared',len(changes),'applied','--apply' in sys.argv)
