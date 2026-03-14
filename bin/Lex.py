import torch
from typing import List, Optional


class Lex:
    """Word-level tokenizer producing span tables over an immutable source buffer.

    A span table is a tensor [num_spans, 3] where each row is (start, end, type):
      - start, end: byte offsets into the source buffer
      - type: integer token ID (vocabulary index)
    """

    def __init__(self, granularity="word"):
        self.granularity = granularity
        self.vocab = {}        # word -> token_id
        self.id_to_word = {}   # token_id -> word

    def build_vocab(self, source: torch.Tensor) -> None:
        """Scan source buffer, extract words, assign token IDs."""
        text = bytes(source.tolist()).decode('utf-8')
        if self.granularity == "word":
            words = text.split()
        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}")
        next_id = 0
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = next_id
                self.id_to_word[next_id] = word
                next_id += 1

    def encode(self, source: torch.Tensor,
               example_offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return span table [num_spans, 3] over source buffer.

        If example_offsets is provided as [N, 2] tensor of (start, end) pairs,
        only spans within those byte ranges are returned.
        """
        text = bytes(source.tolist()).decode('utf-8')
        spans = []
        pos = 0
        for word in text.split():
            start = text.index(word, pos)
            end = start + len(word.encode('utf-8'))
            token_id = self.vocab[word]
            spans.append([start, end, token_id])
            pos = end

        result = torch.tensor(spans, dtype=torch.long)

        if example_offsets is not None:
            mask = torch.zeros(len(spans), dtype=torch.bool)
            for i in range(example_offsets.shape[0]):
                lo, hi = example_offsets[i, 0].item(), example_offsets[i, 1].item()
                row_mask = (result[:, 0] >= lo) & (result[:, 1] <= hi)
                mask |= row_mask
            result = result[mask]

        return result

    def decode(self, source: torch.Tensor, spans: torch.Tensor) -> List[str]:
        """Reconstruct surface strings by slicing source at span offsets."""
        raw = bytes(source.tolist())
        words = []
        for i in range(spans.shape[0]):
            start = spans[i, 0].item()
            end = spans[i, 1].item()
            words.append(raw[start:end].decode('utf-8'))
        return words
