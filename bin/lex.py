import re
import torch
from typing import List, Optional

SENTENCE_SEPARATORS = set('.!?')


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
        text = bytes(source.cpu().numpy()).decode('utf-8')
        if self.granularity == "word":
            words = text.split()
        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}")
        next_id = len(self.vocab)
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = next_id
                self.id_to_word[next_id] = word
                next_id += 1

    def encode(self, source: torch.Tensor,
               example_offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return span table [num_spans, 3] over source buffer.

        Offsets are byte positions into the source tensor (not character
        positions), so multi-byte UTF-8 characters are handled correctly.

        If example_offsets is provided as [N, 2] tensor of (start, end) pairs,
        only spans within those byte ranges are returned.
        """
        raw = bytes(source.cpu().numpy())
        text = raw.decode('utf-8')

        # Build char->byte offset map once, O(n)
        byte_offsets = []
        bi = 0
        for ch in text:
            byte_offsets.append(bi)
            bi += len(ch.encode('utf-8'))
        byte_offsets.append(bi)  # sentinel for end of string

        spans = []
        for m in re.finditer(r'\S+', text):  # O(n) single pass, no re-search
            word = m.group()
            start_char, end_char = m.start(), m.end()
            start_byte = byte_offsets[start_char]
            end_byte = byte_offsets[end_char]
            if word not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[word] = new_id
                self.id_to_word[new_id] = word
            spans.append([start_byte, end_byte, self.vocab[word]])

        result = torch.tensor(spans, dtype=torch.long) if spans else torch.zeros((0, 3), dtype=torch.long)

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
        """Tokenize buf[start:] into spans using character-class scanning.

        Returns:
            list of {'start': int, 'end': int, 'text': str, 'category': str}

        Each token is categorized as one of:
            'WORD'      -- alphabetic runs (including mid-word apostrophes/hyphens)
            'NUMBER'    -- digit runs
            'SEPARATOR' -- sentence-ending punctuation (. ! ?)
            'PUNCT'     -- all other non-whitespace single characters

        Sentence grouping is NOT done here -- that is parse's job.
        Only WORD tokens update self.vocab.
        """
        text = buf[start:]
        if not text.strip():
            return []

        tokens = []
        i = 0
        n = len(text)

        def _add_token(tok_start, tok_end, category):
            tok_text = text[tok_start:tok_end]
            tokens.append({
                'start': start + tok_start,
                'end': start + tok_end,
                'text': tok_text,
                'category': category,
            })
            if category == 'WORD' and tok_text not in self.vocab:
                next_id = len(self.vocab)
                self.vocab[tok_text] = next_id
                self.id_to_word[next_id] = tok_text

        after_separator = True   # start of buffer = treat like after separator (skip leading whitespace)

        while i < n:
            ch = text[i]

            # Whitespace run -- SPACE token only within a sentence (not after SEPARATOR)
            if ch in ' \n\r\t':
                j = i + 1
                while j < n and text[j] in ' \n\r\t':
                    j += 1
                if not after_separator:
                    _add_token(i, j, 'SPACE')
                i = j
                continue

            # Alphabetic run (WORD) -- includes mid-word apostrophes and hyphens
            if ch.isalpha():
                j = i + 1
                while j < n:
                    c = text[j]
                    if c.isalpha():
                        j += 1
                    elif c in ("'", "-", "\u2019") and j + 1 < n and text[j + 1].isalpha():
                        # Mid-word apostrophe/hyphen: include it and the next alpha
                        j += 2
                    else:
                        break
                _add_token(i, j, 'WORD')
                after_separator = False
                i = j
                continue

            # Digit run (NUMBER)
            if ch.isdigit():
                j = i + 1
                while j < n and text[j].isdigit():
                    j += 1
                _add_token(i, j, 'NUMBER')
                after_separator = False
                i = j
                continue

            # Sentence-ending punctuation (SEPARATOR)
            if ch in SENTENCE_SEPARATORS:
                _add_token(i, i + 1, 'SEPARATOR')
                after_separator = True
                i += 1
                continue

            # Everything else: single-character PUNCT
            _add_token(i, i + 1, 'PUNCT')
            after_separator = False
            i += 1

        return tokens
