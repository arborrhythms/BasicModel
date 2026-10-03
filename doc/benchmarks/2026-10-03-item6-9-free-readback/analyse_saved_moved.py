"""Inspect saved moved-case tensors; construct no model and draw no randomness."""
from collections import Counter
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
folder = HERE / 'moved-observations/bf3582592766e765'
data = torch.load(folder / 'inverse-000.pt', weights_only=True, map_location='cpu')
decoded = json.loads((folder / 'decoded.json').read_text())
rows, codes, leaves = data['rows'], data['codes'], data['recovered']
valid = rows >= 0
dot = (leaves[:, :, None] * codes[:, None]).sum(-1)
square = codes.square().sum(-1)
activation = dot / square[:, None].clamp_min(torch.finfo(codes.dtype).tiny)
cosine = torch.nn.functional.cosine_similarity(leaves[:, :, None], codes[:, None], dim=-1)
score = activation * cosine * data['weights'][:, None]
score = score.masked_fill(~valid[:, None], -1e4)
assignment = torch.softmax(torch.cat((score / .1, score.new_zeros(*score.shape[:2], 1)), -1), -1)
choice = score.argmax(-1)
targets, predictions, byte_probabilities = [], [], []
sentences = []
missing = []
for b in range(len(rows)):
    target_words, predicted_words, costs = [], [], []
    for w in torch.where(data['word_valid'][b])[0].tolist():
        matches = torch.where(rows[b] == data['word_rows'][b, w])[0]
        if not len(matches):
            missing.append([b, w]); continue
        target = data['surfaces'][b][int(matches[0])]
        prediction = data['surfaces'][b][int(choice[b, w])]
        targets.append(target); predictions.append(prediction)
        target_words.append(target); predicted_words.append(prediction)
        probabilities = []
        for position, symbol in enumerate(target.encode('utf8') + b'\0'):
            probability = assignment[b, w, -1] / 256
            for k, surface in enumerate(data['surfaces'][b]):
                if not bool(valid[b, k]): continue
                word = surface.encode('utf8') + b'\0'
                if position < len(word) and word[position] == symbol:
                    probability = probability + assignment[b, w, k]
            probabilities.append(float(probability))
        byte_probabilities.extend(probabilities)
        costs.append(float(-torch.tensor(probabilities).clamp_min(1e-6).log().mean()))
    sentences.append(dict(input=decoded['targets'][b], decoded=decoded['decoded'][b],
        target_surfaces=target_words, recovered_surfaces=predicted_words,
        raw_byte_cost_from_saved_values=sum(costs) / len(costs) if costs else None,
        truncated=bool(data['truncated'][b]),
        rule_ids=data['rule_ids'][b][data['rule_valid'][b]].tolist(),
        arities=data['arities'][b][data['rule_valid'][b]].tolist()))
result = dict(scope='Arithmetic on the saved final evaluation only. No model, forward, optimizer, or new random draw. Byte probabilities are reconstructed from saved candidate surfaces, codes, priming and recovered vectors; no derivative is recomputed.',
    sentences=len(sentences), exact=sum(a == b for a, b in zip(decoded['targets'], decoded['decoded'])),
    truncated=int(data['truncated'].sum()), active_word_occurrences=len(targets),
    correct_word_occurrences=sum(a == b for a, b in zip(targets, predictions)),
    target_surfaces=dict(Counter(targets)), predicted_surfaces=dict(Counter(predictions)),
    whitespace_occurrences=sum(x.isspace() for x in targets),
    correct_whitespace_occurrences=sum(a.isspace() and a == b for a, b in zip(targets, predictions)),
    byte_positions=len(byte_probabilities),
    byte_probabilities_below_existing_clamp=sum(x < 1e-6 for x in byte_probabilities),
    missing_own_candidate=missing, details=sentences)
(HERE / 'saved-free-derivation-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'details'}))
