from pathlib import Path
p=Path('bin/Language.py'); s=p.read_text()
a=s.index('    def reverse(self, parent, basis=None,', s.index('class SynthesizeLayer'))
b=s.index('        if self.butterfly:',a)
s=s[:b]+"        if self.meaning_layout is not None and parent.shape[-1] == self.meaning_layout[1]:\n            self.raise_no_inverse('a meaning fold needs its context or the candidate search')\n"+s[b:]
a=s.index('    def reverse(self, parent, *, case_bank=None):',s.index('class CompoundLayer'))
b=s.index('        if case_bank is None:',a)
s=s[:b]+"        if self.meaning_layout is not None and parent.shape[-1] == self.meaning_layout[1]:\n            self.raise_no_inverse('a meaning fold needs its context or the candidate search')\n"+s[b:];p.write_text(s)
p=Path('bin/Layers.py');s=p.read_text();a=s.index('class EqualLayer'); b=s.index('        return Ops._equal_kernel(left, right)',a)
s=s[:b]+"        if self.meaning_layout is not None and left.shape[-1] == self.meaning_layout[1]:\n            return self.compose_blocks(None, left, right)\n"+s[b:];a=s.index('    def reverse(self, parent):',a);b=s.index('        return parent, parent',a)
s=s[:b]+"        if self.meaning_layout is not None and parent.shape[-1] == self.meaning_layout[1]:\n            self.raise_no_inverse('a meaning fold needs its context or the candidate search')\n"+s[b:];p.write_text(s)
p=Path('bin/Models.py');s=p.read_text();a=s.index('    def _reverse_event_loss(');b=s.index('        Kr = ',a)
s=s[:b]+"        # Native reconstruction can return byte identities. This optional\n        # diagnostic is numerical even when both observed surfaces are ints.\n        dtype = torch.promote_types(rev_ev.dtype, fwd_ev.dtype)\n        if not dtype.is_floating_point:\n            dtype = torch.get_default_dtype()\n        rev_ev, fwd_ev = rev_ev.to(dtype), fwd_ev.to(dtype)\n"+s[b:];p.write_text(s)
p=Path('test/test_identity_cube.py');s=p.read_text();s=s.replace('def test_native_synthesis_adds_in_chart_and_keeps_its_inverse():','def test_native_synthesis_centroids_form_and_inverts_meaning():').replace('    from Layers import cube_chart, cube_unchart\n','    from MeaningKernels import LaneKernel\n',1).replace('    layer.configure_identity_domain()\n','    layer.meaning_kernel = LaneKernel(2)\n    layer.configure_identity_domain()\n',1).replace('    a, b = layer.generate(parent)\n','    a, b = layer.reverse_blocks(parent, right)\n',1).replace('    recovered = cube_unchart(cube_chart(parent[..., :2]) - cube_chart(left[..., :2]))\n','    recovered = 2 * parent[..., :2] - left[..., :2]\n',1);p.write_text(s)
p=Path('test/test_meaning_kernels.py');s=p.read_text();s+='''\n\ndef test_direct_paired_reverses_reject_a_retired_split(paired):\n    from Language import SynthesizeLayer, CompoundLayer\n    from Layers import EqualLayer\n    from Exceptions import NoInverse\n    a, b = torch.linspace(.15, .75, 12), torch.linspace(.8, .2, 12)\n    for cls in (SynthesizeLayer, CompoundLayer, EqualLayer):\n        op = cls()\n        result = op.compose(a, b)\n        with pytest.raises(NoInverse, match='candidate search'):\n            op.reverse(result)\n        left, right = op.reverse_blocks(result, b)\n        torch.testing.assert_close(left, a)\n        torch.testing.assert_close(right, b)\n'''
# Use the project's exception owner, verified below before running.
p.write_text(s)
