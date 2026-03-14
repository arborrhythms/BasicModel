import torch
from typing import List, Optional

SENTENCE_SEPARATORS = set('.!?;')


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

    def lex_buffer(self, buf, start=0):
        """Tokenize buf[start:] into word and separator spans.

        Returns:
            list of {'start': int, 'end': int, 'text': str, 'category': str}

        Each token is categorized as 'WORD' or 'SEPARATOR'.
        Sentence grouping is NOT done here — that is parse's job.
        Also updates self.vocab with new words encountered.
        """
        text = buf[start:]
        if not text.strip():
            return []

        tokens = []
        i = 0
        while i < len(text):
            # Skip whitespace
            if text[i] in ' \n\r\t':
                i += 1
                continue

            # Find token end
            j = i
            while j < len(text) and text[j] not in ' \n\r\t':
                j += 1

            raw_token = text[i:j]

            # Check if token ends with separator punctuation
            # e.g. "barks." → WORD "barks" + SEPARATOR "."
            if len(raw_token) > 1 and raw_token[-1] in SENTENCE_SEPARATORS:
                word_part = raw_token[:-1]
                sep_part = raw_token[-1]

                abs_start = start + i
                abs_end = start + i + len(word_part)
                tokens.append({
                    'start': abs_start,
                    'end': abs_end,
                    'text': word_part,
                    'category': 'WORD',
                })
                if word_part not in self.vocab:
                    next_id = len(self.vocab)
                    self.vocab[word_part] = next_id
                    self.id_to_word[next_id] = word_part

                sep_start = abs_end
                sep_end = sep_start + 1
                tokens.append({
                    'start': sep_start,
                    'end': sep_end,
                    'text': sep_part,
                    'category': 'SEPARATOR',
                })
            elif raw_token in SENTENCE_SEPARATORS:
                tokens.append({
                    'start': start + i,
                    'end': start + j,
                    'text': raw_token,
                    'category': 'SEPARATOR',
                })
            else:
                tokens.append({
                    'start': start + i,
                    'end': start + j,
                    'text': raw_token,
                    'category': 'WORD',
                })
                if raw_token not in self.vocab:
                    next_id = len(self.vocab)
                    self.vocab[raw_token] = next_id
                    self.id_to_word[next_id] = raw_token

            i = j

        return tokens
