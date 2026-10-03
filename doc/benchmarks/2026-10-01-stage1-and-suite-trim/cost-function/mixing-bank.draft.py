    def _stage_mixing_reconstruction_bank(self):
        """Snapshot the mixing reading's object candidates and observed bytes.

        This is the eager reading boundary. Object references remain grammar
        references; none of the native WORD identity fields are published.
        Candidate surfaces belong to admitted objects, while scoring targets
        come from the observed units, including words not yet admitted.
        """
        if not self.serial or self._aligned_serial_word_mode():
            return
        isp = self.inputSpace
        active = getattr(isp, '_word_active_mask', None)
        rows = getattr(isp, '_ar_grammar_object_rows', None)
        atoms = getattr(isp, '_ar_grammar_object_atoms', None)
        if not (torch.is_tensor(active) and torch.is_tensor(rows)
                and torch.is_tensor(atoms)):
            return
        isp._finalize_sentence_word_layout(active)
        B, W = active.shape
        owner = self._concept_owner()
        percepts = getattr(self.perceptualSpace, '_forward_input', None) or {}
        texts = percepts.get('word_texts')
        tokens = percepts.get('tokens', ())
        spans = percepts.get('part_spans')
        raw = getattr(self, '_staged_concepts_in', None)
        if torch.is_tensor(spans) and torch.is_tensor(raw):
            spans = spans.detach().cpu().tolist()
            raw = raw.detach().cpu().reshape(B, -1).long().tolist()
        else:
            spans = raw = None
        surfaces = [[b''] * W for _ in range(B)]
        targets = [[b''] * W for _ in range(B)]
        width = 1
        live = active.detach().cpu().tolist()
        for b, row in enumerate(rows.detach().cpu().tolist()):
            for w, concept_row in enumerate(row):
                if not live[b][w]:
                    continue
                spelling = owner.word_surface_for_row(concept_row) if concept_row >= 0 else None
                if spelling:
                    surfaces[b][w] = spelling
                surface = None
                if texts is not None and b < len(texts) and w < len(texts[b]):
                    surface = texts[b][w]
                elif spans is not None and b < len(spans) and w < len(spans[b]):
                    lo, hi = spans[b][w]
                    if 0 <= lo < hi <= len(raw[b]):
                        surface = bytes(raw[b][lo:hi])
                elif b < len(tokens) and w < len(tokens[b]):
                    surface = tokens[b][w]
                if isinstance(surface, str):
                    surface = surface.encode('utf-8')
                if isinstance(surface, bytes):
                    targets[b][w] = surface
                width = max(width, len(surfaces[b][w]), len(targets[b][w]))
        def tensor_bytes(values):
            data = torch.zeros(B, W, width + 1, dtype=torch.long, device='cpu')
            valid = torch.zeros_like(data, dtype=torch.bool)
            for b, row in enumerate(values):
                for w, value in enumerate(row):
                    if value:
                        data[b, w, :len(value)] = torch.tensor(list(value), device='cpu')
                        valid[b, w, :len(value)] = True
            return data.to(rows.device), valid.to(rows.device)
        isp._ar_concept_lookup_rows = rows.clone()
        isp._ar_concept_lookup_atoms = atoms
        isp._ar_concept_lookup_sentence_ids = torch.where(
            rows >= 0, isp._packed_sentence_ids, torch.full_like(rows, -1))
        isp._ar_bank_bytes, isp._ar_bank_valid = tensor_bytes(surfaces)
        isp._ar_target_word_bytes, isp._ar_target_word_mask = tensor_bytes(targets)

