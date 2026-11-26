"""Measure the unchanged first Pi on a restored reader's live form codes."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', default='data/XOR_grammar.xml')
    parser.add_argument('--words', nargs='+', default=['blue', 'hello', 'loving', 'red', 'there', 'world'])
    args = parser.parse_args()
    if len(args.words) < 2 or len(set(args.words)) != len(args.words):
        parser.error('--words needs at least two distinct words')
    from Models import BaseModel
    from data import Data
    from WordIdentity import bytes_of
    from BasicPercepts import least_of_greater
    model, _ = BaseModel.from_config(args.config, data=Data())
    try:
        if not model.load_weights(str(args.checkpoint), require_match=True):
            raise ValueError('could not restore the requested reader')
        ps = model.perceptualSpace
        identity, table = ps.percept_store.identity, ps.form_links
        raw = [bytes_of(word) for word in args.words]
        width = max(map(len, raw))
        data = table.left.new_tensor([list(word) + [0] * (width-len(word)) for word in raw])
        rows = table.read(data, table.left.new_tensor(list(map(len, raw))), admit=False)
        if bool((rows < 0).any()):
            raise ValueError('a requested word has no row in the saved table')
        codes = table.code[rows, :identity.width]
        read = least_of_greater(codes[None], 1 - codes)[:, 0]
        store_rows = rows.new_tensor([identity.word_rows[word] for word in raw])
        actual = ps.subspace.what.lookup_rows(store_rows)[:, :identity.width]
        mask = ~torch.eye(len(raw), dtype=torch.bool, device=read.device)
        result = dict(words=args.words, table_rows=rows.tolist(),
            live_table_read_equal=torch.equal(actual, codes),
            code_min=float(codes.min()), code_max=float(codes.max()),
            own_mean=float(read.diag().mean()), other_mean=float(read[mask].mean()),
            contrast=float(read.diag().mean()-read[mask].mean()), matrix=read.tolist())
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        model.End()


if __name__ == '__main__':
    main()
