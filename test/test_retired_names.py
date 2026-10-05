"""One table of retired interfaces; behavioral checks stay in their own files."""
import importlib
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Module:qualified object, followed by retired attribute names.
ATTRIBUTES = (
    ('Spaces:WholeSpace', ('insert_meta', 'insert_relation', 'taxonomy_children',
        'taxonomy_parent', '_migrate_signed_int_taxonomy', '_record_truth_activations',
        '_record_lbg_pull', '_maybe_split_lbg', 'meta_trust', 'insert_percept',
        'insert_whole', 'reverse_decode')),
    ('Spaces:ConceptualSpace', ('conceptualize_chain', '_chain_missing_concepts',
        'create_joint_concept', '_automatic_joint_concept', 'create_word_object_meta',
        '_automatic_word_object_meta', '_learn_score_children_in_codebook',
        '_learn_score_is_truth_obvious', '_learn_score_resolves_contradiction',
        '_maybe_learn_relation', '_route_learned_relation', '_compute_learn_score',
        'learn_relations_from_stm', '_relation_is_reducible', '_collapse_trust',
        '_sourced_input', '_read_event', '_get_active_input_sibling', '_build_symbol_leg')),
    ('Spaces:PartSpace', ('_maybe_autobind_words', '_sourced_input', '_read_event')),
    ('bin.Spaces:InputSpace', ('expand_masked', 'arir_step', 'getBatch', 'predict',
        'embed_token', 'get_space_embedding', 'get_mask_embedding', '_lexicon')),
    ('bin.Spaces:OutputSpace', ('expand_masked',)),
    ('Spaces:', ('SubwholeSpace',)),
    ('Language:', ('IdeaSubSpace', 'binary_tiling_soft_dp', 'binary_tiling_viterbi', 'compact_hard',
        'compact_soft', 'comparator_dp_kl', 'UnaryStructuredLayer',
        'BinaryStructuredReductionLayer', '_normalize_grammar_dict', 'Chart',
        '_parse_cfg_lines')),
    ('Language:Grammar', ('load_from_cfg', '_parse_cfg_lines')),
    ('Language:SymbolSubSpace', ('stm_residual', 'stm_residual_microbatch', 'arm_stm')),
    ('Models:BasicModel', ('_set_superposition_temperature', '_intersentence_seed',
        '_consume_intersentence_seed')),
    ('Layers:GrammarLayer', ('emit', 'bind_marker', 'canonical_marker', 'bound_markers')),
    ('space_carrier:SpaceCarrierMixin', ('mark_codebook_parameters_changed', 'mark_codebook_structure_changed')),
    ('Layers:', ('RelativeTruthStore', 'relu_chart', 'relu_inject',
        'membership_inject', 'inject_membership', 'to_membership_relu')),
    ('Layers:Ops', ('relu_chart', 'relu_inject', 'membership_inject',
        'inject_membership', 'to_membership_relu')),
)
SOURCES = (
    ('Language:OperationSelectionLayer', ('op_space_role_idx', 'position_space_role'), False),
    ('Language:SymbolSpace.forward_concept_to_symbol',
        ('_model_symbolSpace', '_relation_store', 'wholeSpace', 'WholeSpace'), True),
    ('bin.Spaces:InputSpace.__init__', ('_peer_embedding',), False),
    ('bin.Spaces:InputSpace.forward', ('_raw_input',), False),
    ('bin.Spaces:PartSpace.embed_stem', ('_raw_input', 'vocab.forward'), False),
    ('Layers:LiftLayer', ('PartSpace.sigma', 'ConceptualSpace.pi'), False),
    ('Layers:LowerLayer', ('PartSpace.sigma', 'ConceptualSpace.pi'), False),
    ('Spaces:PartSpace.__init__', ('self.conceptualSpace_ref = None',), False),
    ('Spaces:ConceptualSpace.__init__', ('self.wholeSpace_ref = None',
        'self.perceptualSpace_ref = None', 'self.subwholeSpace_ref = None'), False),
)
SIGNATURES = (
    ('Spaces:ConceptualSpace.bind_streams', ('seed_payload',)),
    ('Spaces:ConceptualSpace.__init__', ('subsymbolic_widen_dim',)),
)
ARCHITECTURE_KEYS = ('nInput', 'nPercepts', 'nConcepts', 'nSymbols', 'nOutput',
    'inputDim', 'perceptDim', 'conceptDim', 'symbolDim', 'outputDim',
    'perceptPassThrough', 'symbolPassThrough', 'perceptPrototypes',
    'conceptPrototypes', 'perceptHasAttention', 'conceptHasAttention')
GRAMMAR_TOKENS = frozenset(('and', 'AND', 'NP', 'NP3', 'NP4', 'NP34', 'NP345',
    'AP', 'AP4', 'VP', 'VP1', 'N3', 'DET', 'ADV', 'ADJ', 'P', 'V1', 'S3', 'S4',
    'S5', 'S34', 'S45', 'S345', 'MP1', 'PP', 'REL_T', 'ABS_T', 'CONJ_L45',
    'CONJ_R45', 'DISJ_L45', 'DISJ_R45', 'CONJ_L3', 'CONJ_R3', 'DISJ_L3',
    'DISJ_R3', 'QLEFT_NP3', 'QRIGHT_AP', 'QRIGHT_NP3', 'QRIGHT_S34', 'NP_EQ3',
    'NP_EQ4', 'NP_EQ345', 'S_PART34', 'QLEFT_PART34'))
# Keep instance checks on instances, including every conceptual stage.
MODEL_ATTRIBUTES = (
    ('radix', 'perceptualSpace', ('wholeSpace_ref',)),
    ('parallel', 'conceptualSpaces.*', ('sigma_in', 'sigma_cs', 'sigma')),
    ('plain', 'conceptualSpace', ('sigma_percept', 'sigma_percept_1',
        'sigma_percept_2', '_sigma_percept_reverse', 'sigma', 'forwardPi', 'reversePi')),
    ('plain', 'perceptualSpace', ('pi', 'pi_input', 'pi_concept')),
    ('plain', 'wholeSpace', ('sigma', 'forwardSigma', 'reverseSigma', '_sigma_reverse',
        'insert_paired_word', 'mark_word_atom')),
    ('plain', 'outputSpace', ('_piLayer',)),
    ('plain', '', ('_ws_cache', '_cs_cache')),
    ('text', 'wholeSpace', ('insert_paired_word', 'mark_word_atom')),
    ('no_eos', 'inputSpace', ('batch_advances_sentence',)),
    ('xor', 'wholeSpace', ('use_stack_router', '_stack_route_forward')),
    ('enum', '', ('perfect_reconstruction',)),
)


def _resolve(address):
    module, path = address.split(':', 1)
    value = importlib.import_module(module)
    for field in filter(None, path.split('.')):
        value = getattr(value, field)
    return value


def _models():
    # These are the original fixture constructors. Identical MM_xor_loopback
    # construction is shared once across its attribute-only checks.
    from test_autobind_from_cs import _make_radix_model
    from test_cs_reentrancy import _make_model
    from test_cs_stm_bookkeeping import _make_plain_model
    from test_lexicon_ownership import _build_text_model
    from test_no_eos_sync import _model
    from test_subspace_what_stm_contract import _xor_model
    from test_conceptual_recurrence import _build
    return dict(radix=_make_radix_model, parallel=_make_model,
        plain=_make_plain_model, text=_build_text_model, no_eos=_model,
        xor=_xor_model.__wrapped__, enum=lambda: _build('MM_20M_xor.xml'))


def test_retired_names_remain_absent():
    import torch
    import Language
    import Layers
    import Models
    from util import TheXMLConfig

    for address, names in ATTRIBUTES:
        owner = _resolve(address)
        for name in names:
            assert not hasattr(owner, name), (address, name)
    for address, names, strip_doc in SOURCES:
        source = inspect.getsource(_resolve(address))
        if strip_doc:
            head, _, rest = source.partition('"""')
            source = head + rest.partition('"""')[2]
        for name in names:
            assert name not in source, (address, name)
    for address, names in SIGNATURES:
        parameters = inspect.signature(_resolve(address)).parameters
        for name in names:
            assert name not in parameters, (address, name)

    instances = (
        (Layers.ConceptualCombine(content_dim=6, naive=False, sigma_pi_mode='full'),
            ('reverse_dropped', 'aug_dim')),
        (Layers.ConceptAllocator(), ('chain_idx', 'joint')),
        (Layers.SparseLayer(4, 3), ('ensure_row_key', 'embed_pair', 'constituents',
            'row_is_identity', 'assign_row', 'hebbian_strengthen_row')),
        (Language.RuleCodebook(num_rules=1), ('forward_to_parent_what',)),
        (Layers.TruthLayer(nDim=8, max_truths=16), ('should_store',)),
        (Layers.BracketExpectation(n_symbols=4, max_depth=2, n_dim=3, p=5, q=2,
            concept_dim=6, batch=1), ('prime', 'cast')),
        (Layers.ChunkLayer(nDim=8, bpe=True, n_vectors=1024, word_learning=2),
            ('split', 'merge', 'threshold', 'score_pair', 'encode', 'decode', 'should_merge')),
    )
    for owner, names in instances:
        for name in names:
            assert not hasattr(owner, name), (type(owner).__name__, name)
    assert hasattr(instances[-1][0], 'merges')
    for cls in (Layers.PiLayer2, Layers.SigmaLayer2):
        instance = cls(4, 4)
        for name in ('_to_mult', '_from_mult'):
            assert name not in cls.__dict__
            assert not hasattr(instance, name)
        for method in ('forward', 'reverse', 'compose'):
            source = inspect.getsource(getattr(cls, method))
            for name in ('_to_mult', '_from_mult', 'atanh', 'tanh'):
                assert name not in source, (cls.__name__, method, name)
    lift = Layers.LiftLayer(wholeSpace=None, perceptualSpace=None)
    assert getattr(lift, 'raw_gate', None) is None
    assert 'raw_gate' not in dict(lift.named_parameters())
    from embed import WordVectors
    words = WordVectors(torch.randn(4, 8), ['a', 'b', 'c', 'd'])
    assert not hasattr(words, 'tie_to_codebook')
    assert words._tied_param_getter is None
    assert hasattr(Layers.Ops, 'eval_chart') and hasattr(Layers.Ops, 'eval_chart_inv')
    assert hasattr(Language.SymbolSpace, 'forward_concept_to_symbol')

    factories, models = _models(), {}
    try:
        for key, path, names in MODEL_ATTRIBUTES:
            if key not in models:
                # Reproduce pytest's former per-case singleton reset.
                Language.TheGrammar._configured = False
                TheXMLConfig._requirements.clear()
                models[key] = factories[key]()
            values = [models[key]]
            for field in filter(None, path.split('.')):
                values = ([item for value in values for item in value] if field == '*'
                          else [getattr(value, field) for value in values])
            for value in values:
                for name in names:
                    assert not hasattr(value, name), (key, path, name)
        assert callable(getattr(models['enum'], 'reverseReconstruct', None))
        assert not isinstance(vars(models['enum']).get('reconstruct'), str)
    finally:
        for model in models.values():
            model.End()

    cfg = Models.BaseModel.load_config(str(ROOT / 'data/model.xml'))
    for name in ARCHITECTURE_KEYS:
        assert name not in cfg['architecture'], name
    cfg = Models.BaseModel.load_config(str(ROOT / 'data/XOR_grammar.xml'))
    assert 'routerKind' not in cfg['SymbolSpace']
    for path in (ROOT / 'data/complete.grammar', ROOT / 'test/fixtures/transitional_pos.grammar'):
        grammar = Language.Grammar()
        grammar.load_from_grammar_file(str(path))
        for rule in grammar.rules:
            tokens = [s.strip() for s in str(rule.lhs).split(',')] + list(rule.rhs_symbols or ())
            for token in tokens:
                if path.name == 'complete.grammar':
                    assert token not in GRAMMAR_TOKENS, (token, rule.canonical)
                assert not token.endswith('_MARK'), (token, rule.canonical)
        for name in ('copy', 'swap'):
            assert name not in {rule.method_name for rule in grammar.rules if rule.method_name}
    for name in ('grammar2.cfg', 'grammar_legacy.cfg'):
        assert not (ROOT / 'data' / name).exists(), name
