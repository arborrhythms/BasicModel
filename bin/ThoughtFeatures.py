"""Bounded input features for the ordinary grammar controller's MLP.

No interpreter, learned policy, memory or executor lives here. References are
alpha-renamed before categorical metadata is encoded: allocator magnitudes and
namespace spellings cannot supply arithmetic or cross-row information.
"""
import json

import torch
from torch.nn import functional as F

MODES = ('assertive', 'interrogative', 'unspecified')
METADATA_BYTES = 64
METADATA_WIDTH = 4 + 2 * (9 * METADATA_BYTES + 1)


def context_width(width):
    return 9 * width + 15 + 3 * METADATA_WIDTH + 2 * (3 * width + 10) + 3 * width + 7


def semantic_metadata(meanings):
    """Mode, polarity and bounded, ordered binding/scope categorical bytes.

    Bytes are eight independent binary categories plus an occupancy bit, not
    a scalar ordinal. Each field has an explicit overflow bit. Native typed
    references become local aliases, seeded by the nine canonical roles; this
    retains binding equality and structure without exposing address values.
    """
    aliases = {}

    def address(value):
        if value not in aliases:
            aliases[value] = len(aliases)
        return ('reference', value[0], aliases[value])

    def freeze(value, depth=0):
        if depth > 16:
            return ('depth_limit',)
        if isinstance(value, tuple):
            if (value and isinstance(value[0], str)
                    and value[0] in ('sym', 'ltm', 'thought', 'constituent')):
                return address(value)
            return tuple(freeze(item, depth + 1) for item in value[:64])
        return value

    for meaning in meanings:
        for ref in meaning.role_refs:
            if ref is not None:
                address(ref)

    def field(value, like):
        raw = json.dumps(freeze(value), ensure_ascii=False,
                         separators=(',', ':')).encode('utf-8')
        values = like.new_zeros(METADATA_BYTES, 9)
        for index, byte in enumerate(raw[:METADATA_BYTES]):
            values[index, :8] = like.new_tensor([(byte >> bit) & 1 for bit in range(8)])
            values[index, 8] = 1
        return torch.cat((values.reshape(-1), like.new_tensor([len(raw) > METADATA_BYTES])))

    return tuple(torch.cat((meaning.roles.new_tensor(
        [meaning.mode == mode for mode in MODES] + [meaning.polarity]),
        field(meaning.bindings, meaning.roles), field(meaning.scope, meaning.roles)))
        for meaning in meanings)


def attend_meanings(query, records, *, incomplete=False):
    """Attention over detached, actually observed full role payloads.

    Records contain (meaning, signed evidence). The query remains live; no
    desired answer or current training target is accepted by this interface.
    """
    width = int(query.roles.shape[-1])
    if not records:
        empty = query.roles.new_zeros(3 * width + 10)
        empty[-1] = float(incomplete)
        return empty
    keys, values = [], []
    for meaning, trust in records:
        if meaning.roles.shape[-1] != width:
            raise ValueError('thought memory and request must have the same full width')
        roles = meaning.roles.detach().to(query.roles)
        mask = meaning.role_mask.to(roles)
        roles = roles * mask[:, None]
        keys.append(roles.reshape(-1))
        values.append(torch.cat((roles.reshape(-1), mask, roles.new_tensor(
            [meaning.mode == mode for mode in MODES]
            + [meaning.polarity, float(trust), 1., float(incomplete)]))))
    keys = torch.stack(keys)
    q = (query.roles * query.role_mask.to(query.roles)[:, None]).reshape(-1)
    scores = F.normalize(keys, dim=-1) @ F.normalize(q, dim=0)
    return torch.softmax(scores, dim=0) @ torch.stack(values)
