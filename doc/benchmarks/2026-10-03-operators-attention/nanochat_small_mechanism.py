"""Three unchanged manifest items exercise the evaluator, not the item-4 gate."""
import json
from pathlib import Path
import torch
import eval_nanochat_grammar as gate


def test_small_word_evaluator_mechanism():
    config = 'data/MM_grammar_wording.xml'
    model, data = gate.build_eval_model(config, autoload=False)
    manifest = gate.load_manifest(gate.DEFAULT_MANIFEST)
    result = gate.score_manifest(model, data, manifest, limit=3)
    assert [row['id'] for row in result['items']] == [row['id'] for row in manifest['items'][:3]]
    assert result['metrics']['items'] == 3 and result['metrics']['choices'] == 16
    for row in result['items']:
        assert len(row['intact_scores']) == len(row['shuffled_scores']) == 16
        assert torch.isfinite(torch.tensor(row['intact_scores'] + row['shuffled_scores'])).all()
        assert 1 <= row['intact_rank'] <= 16 and 1 <= row['shuffled_rank'] <= 16
        assert row['intact_prediction_steps'] > 0 and row['shuffled_prediction_steps'] > 0
    result.update(purpose='mechanism only; gate deferred to item 4 trained checkpoint',
        config=config, manifest_sha256=manifest['metadata']['item_sha256'],
        seed='unchanged XML seed 931', training_updates=0,
        parameters=sum(parameter.numel() for parameter in model.parameters()))
    (Path(__file__).parent / 'nanochat-small-mechanism.json').write_text(json.dumps(result, indent=2) + '\n')
