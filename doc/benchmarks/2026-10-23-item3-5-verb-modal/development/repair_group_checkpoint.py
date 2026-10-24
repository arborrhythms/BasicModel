from pathlib import Path
p=Path('bin/MeaningKernels.py');s=p.read_text();s=s.replace('''    def get_extra_state(self):
        return dict(version=1, used=self.used, assignments=dict(self.assignments),
                    witnesses={key: tuple(value) for key, value in self.witnesses.items()})

    def set_extra_state(self, state):
        if (state.get('version')''','''    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state(dict(version=1, used=self.used, assignments=dict(self.assignments),
                    witnesses={key: tuple(value) for key, value in self.witnesses.items()}))

    def prepare_checkpoint(self, state, prefix):
        # The first swap freeze wrote the host dictionary directly. Keep
        # that readable while honoring the model's tensor-only envelope.
        from IndependentComponents import _pack_state
        key = prefix + '_extra_state'
        if key in state and not torch.is_tensor(state[key]):
            state[key] = _pack_state(state[key])

    def set_extra_state(self, state):
        from IndependentComponents import _unpack_state
        state = _unpack_state(state)
        if (state.get('version')''');p.write_text(s)
p=Path('bin/Models.py');s=p.read_text().replace('''        from PartsOfSpeech import PartsOfSpeech
        component_metadata = set()''','''        from PartsOfSpeech import PartsOfSpeech
        from MeaningKernels import VerbGroups
        component_metadata = set()''').replace('''            if isinstance(module, ColumnDictionary):
                module.prepare_checkpoint(state, name + '.')
            if isinstance(module, (ColumnDictionary, IndependentComponents, ReferenceClose, PartsOfSpeech)):''','''            if isinstance(module, (ColumnDictionary, VerbGroups)):
                module.prepare_checkpoint(state, name + '.')
            if isinstance(module, (ColumnDictionary, IndependentComponents, ReferenceClose, PartsOfSpeech, VerbGroups)):''');p.write_text(s)
p=Path('test/test_structural_checkpoint.py');s=p.read_text();s+='''\n\ndef test_verb_group_metadata_survives_native_weight_restore(tmp_path):
    from IndependentComponents import _unpack_state
    from MeaningKernels import MeaningAction
    source = _model_with(types.SimpleNamespace(_concept_allocator=_allocator()), types.SimpleNamespace())
    source.action = MeaningAction(8, capacity=2)
    for occurrence in range(3):
        source.action.groups.observe(7, occurrence, (.2, .3), .2)
    source.action.groups.observe(8, 4, (.2, .3), .2)
    path = tmp_path / 'verb-groups.ckpt'
    source.save_weights(path)
    target = _model_with(types.SimpleNamespace(_concept_allocator=_allocator()), types.SimpleNamespace())
    target.action = MeaningAction(8, capacity=2)
    assert target.load_weights(path, require_match=True)
    assert _unpack_state(target.action.groups.get_extra_state()) == _unpack_state(source.action.groups.get_extra_state())
    # The first swap freeze's dictionary is a supported input envelope.
    saved = torch.load(path, weights_only=False)
    saved['state_dict']['action.groups._extra_state'] = _unpack_state(saved['state_dict']['action.groups._extra_state'])
    torch.save(saved, path)
    assert target.load_weights(path, require_match=True)
    assert _unpack_state(target.action.groups.get_extra_state()) == _unpack_state(source.action.groups.get_extra_state())
''';p.write_text(s)
p=Path('doc/plans/2026-10-23-verb-and-modal.md');s=p.read_text();s+='''\n## 11. Lesson restore repair (Codex, 2026-10-23)

The repaired swap/verb source `d3450e16e8` passed every scoped mechanism
group, both decomposition arms and XOR. Its original warm-up completed
76 presentations and 38 paired updates. Stage 3 then failed **before its
first presentation**: `VerbGroups.get_extra_state` returned a dictionary,
where the native weight preflight and the lesson's exact-retention adapter
require tensor envelopes. The failed restore, initial state and warm-up
are retained on that source; this is not a failed behavior gate.

Use the existing `_pack_state`/`_unpack_state` envelope for the group's
host metadata and declare it variable metadata at native restore, just
like occupation counts. The initial dictionary envelope remains readable
through the module's checkpoint preparation. No learned tensor, group
assignment, witness, lesson budget, seed or assertion changes. A native
save/load regression checks a minted group and a pending fit witness,
including the first freeze's envelope. Regenerate the warm-up and resume
the original lesson sequence on the repaired frozen source. This is a
§39 persistence repair, not a rescue of a learning result.
''';p.write_text(s)
