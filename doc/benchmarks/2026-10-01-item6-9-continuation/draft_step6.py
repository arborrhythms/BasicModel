"""Prepare, but do not apply, the next ordered source patch."""
from pathlib import Path
import difflib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
changes = {}

def substitute(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new, 1)

path = 'bin/Language.py'
old = (ROOT / path).read_text()
new = substitute(old, '"""Soft candidate reconstruction using the selected compose kernel.',
                 '"""Least-residual hard pair with the soft candidate gradient.')
new = substitute(new, '''        left = (older * weights[..., None]).sum((1, 2))
        right = (newer * weights[..., None]).sum((1, 2))
        return left, right, active.any(-1)
''', '''        soft_left = (older * weights[..., None]).sum((1, 2))
        soft_right = (newer * weights[..., None]).sum((1, 2))
        selected = residual.masked_fill(~allowed, torch.inf).flatten(1).argmin(-1)
        gather = selected[:, None, None].expand(B, 1, D)
        hard_left = older.reshape(B, K * K, D).gather(1, gather).squeeze(1)
        hard_right = newer.reshape(B, K * K, D).gather(1, gather).squeeze(1)
        available = active.any(-1)
        hard_left = torch.where(available[:, None], hard_left, 0.)
        hard_right = torch.where(available[:, None], hard_right, 0.)
        left = hard_left.detach() + (soft_left - soft_left.detach())
        right = hard_right.detach() + (soft_right - soft_right.detach())
        return left, right, available
''')
changes[path] = old, new

path = 'bin/Models.py'
old = (ROOT / path).read_text()
new = substitute(old, '''                               end_slots=None, end_depth=None, sentence_slot=None):
        """Reconstruct completed sentences with bounded compiled traversals.''',
'''                               end_slots=None, end_depth=None, sentence_slot=None,
                               *, candidate_basis=None, keep_ideas=None):
        """Reconstruct completed sentences with bounded compiled traversals.''')
new = substitute(new, '''        if not torch.compiler.is_compiling():
            self._validate_reconstruction_bank()
        # Targets and inverse witnesses''', '''        if candidate_basis is None and not torch.compiler.is_compiling():
            self._validate_reconstruction_bank()
        # Targets and inverse witnesses''')
new = substitute(new, '''        keep_ideas = bool(self.reconstruct_in_loop or getattr(self, "_recon_keep_ideas", False))
''', '''        keep_ideas = (bool(self.reconstruct_in_loop or getattr(self, "_recon_keep_ideas", False))
                      if keep_ideas is None else bool(keep_ideas))
''')
new = substitute(new, '''        byte_ready, bytes_bwp, valid_bwp = self._byte_tables(B, W)   # the words' bytes
        snap_ready, bank_n, bank_bytes, bank_valid = self._snapshot_tables(reference)
        basis, basis_valid = self._reconstruction_basis_snapshot(reference)
        bank_sentence_ids = getattr(isp, "_ar_concept_lookup_sentence_ids", None)
        if not (torch.is_tensor(bank_sentence_ids) and bank_sentence_ids.shape == basis_valid.shape):
            raise RuntimeError("reconstruction candidates lost their sentence ownership")
        bank_sentence_ids = bank_sentence_ids.detach().to(dev).clone()
        basis_limit = int(self.reconstruction_basis_limit)
        byte_ready = bool(byte_ready and snap_ready)
''', '''        if candidate_basis is None:
            byte_ready, bytes_bwp, valid_bwp = self._byte_tables(B, W)
            snap_ready, bank_n, bank_bytes, bank_valid = self._snapshot_tables(reference)
            basis, basis_valid = self._reconstruction_basis_snapshot(reference)
            bank_sentence_ids = getattr(isp, "_ar_concept_lookup_sentence_ids", None)
            if not (torch.is_tensor(bank_sentence_ids) and bank_sentence_ids.shape == basis_valid.shape):
                raise RuntimeError("reconstruction candidates lost their sentence ownership")
            bank_sentence_ids = bank_sentence_ids.detach().to(dev).clone()
            byte_ready = bool(byte_ready and snap_ready)
        else:
            # Evaluation may search a known numerical vocabulary without
            # manufacturing native WORD symbols or a byte-training bank.
            basis, basis_valid = (value.detach().clone() for value in candidate_basis)
            bank_sentence_ids = torch.full_like(basis_valid, -1, dtype=torch.long)
            bank_n = basis
            bank_bytes = torch.zeros(*basis.shape[:2], 1, device=dev, dtype=torch.long)
            bank_valid = torch.zeros_like(bank_bytes, dtype=torch.bool)
            bytes_bwp = torch.zeros(B, W, 1, device=dev, dtype=torch.long)
            valid_bwp = torch.zeros_like(bytes_bwp, dtype=torch.bool)
            byte_ready = False
        basis_limit = int(self.reconstruction_basis_limit)
''')
new = substitute(new, '''                basis=basis, basis_valid=basis_valid & (bank_sentence_ids == sentence[:, None]),
''', '''                basis=basis, basis_valid=basis_valid & (
                    (bank_sentence_ids == sentence[:, None]) | (bank_sentence_ids < 0)),
''')
new = substitute(new, '''            left, right, unavailable = language.reverse_binary_step(
                top, local_b, valid, ref, inverses=inverses, reference_side=side,
''', '''            if candidate_basis is not None:
                # The read-back measurement gets no original leaf witness.
                ref = torch.zeros_like(ref)
                side = (torch.zeros_like(valid), torch.zeros_like(valid))
            left, right, unavailable = language.reverse_binary_step(
                top, local_b, valid, ref, inverses=inverses, reference_side=side,
''')
method = '''    @_sentence_query_mask
    def reconstruct_grammar_sentence(self, state, sid, active):
        """Read an open sentence's concluded state through its grammar inverse.

        This eager boundary must precede disposal of the reading's operation
        record. Only decoded words and availability leave this call. Candidate
        values are the known object vocabulary; neither source text nor an
        original leaf witness enters the decoder. The ordinary native
        reconstruction byte objective is unchanged.
        """
        self._publish_sentence_scratch(state)
        reference = self._tensor_pushed_ideas
        B, W, D = reference.shape
        owner = self._concept_owner()
        rows = sorted({row for identity in owner.definitions.object_ids
                       if (row := owner._csw_row_of(identity)) is not None
                       and owner.word_surface_for_row(row) is not None})
        if not rows:
            return (None,) * B, active.clone()
        ids = torch.tensor(rows, device=reference.device, dtype=torch.long)
        atoms = owner.similarity_codebook.lookup_rows(ids).detach().to(reference)
        if atoms.shape[-1] != D:
            raise ValueError('grammar reconstruction requires full object codes')
        basis = atoms[None].expand(B, -1, -1)
        valid = torch.ones(basis.shape[:2], dtype=torch.bool, device=reference.device)
        lang = state[1]
        end = lang[13][:, sid].reshape(B, 3, D)
        result = self._reconstruct_sentences(
            lang[9][:, sid], reference, lang[13], lang[14], end, lang[14][:, sid],
            torch.tensor(sid, device=reference.device),
            candidate_basis=(basis, valid), keep_ideas=True)
        scope = self._sentence_scope(sid) & active[:, None]
        lengths = scope.sum(-1)
        # Compact the recovered words, not their retained references. The
        # lexical inverse competes over every owned vocabulary spelling.
        positions = torch.arange(W, device=reference.device).expand(B, W)
        order = torch.argsort(positions + (~scope).long() * W, dim=1)
        words = result[0].gather(1, order[..., None].expand(B, W, D))
        texts = self._generated_word_text(words, lengths)
        unavailable = result[3] & active
        return texts, unavailable

'''
new = substitute(new, '    def _reconstruct_sentence_traversal(self, S, reference, **_unused):\n',
                 method + '    def _reconstruct_sentence_traversal(self, S, reference, **_unused):\n')
changes[path] = old, new

path = 'test/test_explicit_dimensions.py'
old = (ROOT / path).read_text()
new = substitute(old, '''    from Models import ModelFactory
    results = ModelFactory.run("data/XOR_grammar.xml")
    return results[0][2]  # (name, rCorrect, model)
''', '''    from unittest.mock import patch
    from Models import BasicModel, ModelFactory
    commit = BasicModel._commit_sentence
    def capture(model, state, sid, active, *args):
        if not model._sentence_training:
            # Observe the existing evaluation before its temporary trace is
            # discarded. This neither advances the clock nor keeps the trace.
            with torch.no_grad():
                texts, unavailable = model.reconstruct_grammar_sentence(state, sid, active)
            model._grammar_gate_reconstructions = texts
            model._grammar_gate_unavailable = unavailable.detach().cpu().tolist()
        return commit(model, state, sid, active, *args)
    with patch.object(BasicModel, '_commit_sentence', capture):
        results = ModelFactory.run("data/XOR_grammar.xml")
    return results[0][2]  # (name, rCorrect, model)
''')
start = new.index('class TestXorGrammarReconstruction(unittest.TestCase):')
end = new.index('\n\nif __name__ == "__main__":', start)
new = new[:start] + '''class TestXorGrammarReconstruction(unittest.TestCase):
    """All four understandings return their own words, allowing transpositions."""

    @unittest.skipIf(not _RUN_SLOW, "slow (~65s end-to-end XOR_grammar train) -- set RUN_SLOW=1")
    def test_piecewise_overall_at_least_50_pct(self):
        # Keep the historical selector; the decided contract is now all four.
        from collections import Counter
        model = _run_xor_grammar_in_process()
        recon_texts = model._grammar_gate_reconstructions
        test_input, _ = model.inputSpace.getTestData()
        self.assertEqual(len(test_input), 4)
        self.assertEqual(len(recon_texts), 4)
        self.assertFalse(any(model._grammar_gate_unavailable),
                         'every sentence must have an available grammar inverse')
        pairs = [(model._bytes_to_text(original).replace("\\x00", " "), recovered)
                 for original, recovered in zip(test_input, recon_texts)]
        perfect = sum(recovered is not None and Counter(original.split()) ==
                      Counter(recovered.replace("\\x00", " ").split())
                      for original, recovered in pairs)
        self.assertEqual(perfect, 4,
            f"XOR_grammar reconstruction: {perfect}/4 word multisets recovered; {pairs}")
''' + new[end:]
changes[path] = old, new

patch = []
for path, (old, new) in changes.items():
    patch.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                    fromfile='a/' + path, tofile='b/' + path))
(HERE / 'step6-preview.patch').write_text(''.join(patch))
print('Prepared step6-preview.patch; candidate source was not modified.')
