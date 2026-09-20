"""Learned alignment between owned word sequences and grammatical meanings.

This is a language parameter block, not a thought executor or memory owner.
Native addresses label supervision and output selections; only full-width
word/role payloads enter the networks. No natural spelling is declared here.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from Meaning import ConceptualMeaning


@dataclass(frozen=True)
class GeneratedMeaning:
    words: torch.Tensor
    selections: tuple
    truncated: bool


class LinguisticMeaningCodec(nn.Module):
    """Supervised compose alignment and a conditional, copying word decoder."""

    def __init__(self, operation_ids, word_rows, word_values, *, hidden=48):
        super().__init__()
        if (word_values.ndim != 2 or word_values.shape[0] != len(word_rows)
                or len(set(word_rows)) != len(word_rows) or not len(word_rows)):
            raise ValueError("language vocabulary needs unique WORD rows and full-width values")
        self.operation_ids = tuple(operation_ids)
        self.hidden = int(hidden)
        self.width = int(word_values.shape[1])
        import hashlib
        self.register_buffer('schema', torch.tensor([1, self.hidden], dtype=torch.long))
        self.register_buffer('operation_keys', torch.tensor([
            int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'big') >> 1
            for name in self.operation_ids], dtype=torch.long))
        self.register_buffer('word_rows', torch.tensor(word_rows, dtype=torch.long))
        self.register_buffer('word_values', word_values.detach().clone())
        self.encoder = nn.GRU(self.width, self.hidden, batch_first=True, bidirectional=True)
        self.role_head = nn.Linear(2 * self.hidden, 2)
        self.operation_encoder = nn.Sequential(nn.Linear(self.width, 2 * self.hidden), nn.Tanh())
        self.operation_attention = nn.Linear(2 * self.hidden, 1)
        self.operation_head = nn.Linear(2 * self.hidden, len(self.operation_ids) + 1)
        self.mode_head = nn.Linear(2 * self.hidden, 2)
        self.decoder_start = nn.Linear(self.width + 4, self.hidden)
        self.decoder = nn.GRU(self.width, self.hidden, batch_first=True)
        # Vocabulary rows plus canonical NP1, NP2 copy and end of sequence.
        self.copy_codes = nn.Parameter(torch.randn(3, self.width) * .02)
        self.word_head = nn.Linear(self.hidden, len(word_rows) + 3)
        with torch.no_grad():
            self.operation_head.weight.zero_()
            self.operation_head.bias.zero_()
            self.operation_head.bias[-1] = 3.0  # untrained language is unknown

    def encode(self, programs):
        lengths = [len(program.leaves) for program in programs]
        if not lengths or min(lengths) < 1 or max(lengths) > 128:
            raise ValueError("linguistic meanings require 1–128 owned word leaves")
        values = nn.utils.rnn.pad_sequence(
            [program.leaves.to(self.word_values) for program in programs], batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(values, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        valid = torch.arange(encoded.shape[1], device=encoded.device)[None] < torch.tensor(
            lengths, device=encoded.device)[:, None]
        pooled = (encoded * valid[..., None]).sum(1) / torch.tensor(lengths, device=encoded.device)[:, None]
        pointers = self.role_head(encoded).transpose(1, 2).masked_fill(~valid[:, None], -torch.inf)
        lexical = self.operation_encoder(values)
        attention = self.operation_attention(lexical).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(-1)
        operation = (lexical * attention[..., None]).sum(1)
        return self.operation_head(operation), pointers, self.mode_head(pooled)

    def _start(self, meaning):
        features = torch.cat((meaning.roles[1], meaning.role_mask.to(meaning.roles),
                              meaning.roles.new_tensor([float(meaning.polarity)])))
        return torch.tanh(self.decoder_start(features.to(self.word_values)))

    def _embeddings(self):
        return torch.cat((self.word_values, self.copy_codes))

    def loss(self, examples, registry):
        programs = tuple(example[0] for example in examples)
        operations, pointers, modes = self.encode(programs)
        targets, losses, decode_targets, decode_states = [], [], [], []
        vocabulary = self.word_rows.tolist()
        count = len(vocabulary)
        for row, (program, meaning, realization) in enumerate(examples):
            if meaning is None:
                targets.append(len(self.operation_ids))
                continue
            meaning = meaning.detached()
            signature = registry.signature_for(ConceptualMeaning(
                meaning.roles, meaning.role_mask, **dict(meaning.metadata(), mode='interrogative')))
            targets.append(self.operation_ids.index(signature.operation.semantic_id))
            if tuple(signature.occupied_roles) != ('I1', 'I2'):
                raise ValueError("linguistic alignment currently requires two complete concept operands")
            native = program.concept_ids.tolist()
            for position, slot in enumerate((0, 2)):
                reference = meaning.role_refs[slot]
                matches = [i for i, identifier in enumerate(native)
                           if reference == ('sym', identifier)]
                if not matches:
                    raise ValueError("linguistic supervision has no owned operand occurrence")
                losses.append(-torch.logsumexp(F.log_softmax(pointers[row, position], -1)[matches], 0))
            losses.append(F.cross_entropy(modes[row:row + 1], modes.new_tensor(
                [meaning.mode == 'interrogative'], dtype=torch.long)))
            if realization is not None:
                sequence = []
                for word_row, identifier in zip(realization.word_rows.tolist(), realization.concept_ids.tolist()):
                    copies = [i for i, slot in enumerate((0, 2))
                              if meaning.role_refs[slot] == ('sym', identifier)]
                    if copies:
                        sequence.append(count + copies[0])
                    elif word_row in vocabulary:
                        sequence.append(vocabulary.index(word_row))
                    else:
                        raise ValueError("realization contains an unconfigured WORD row")
                sequence.append(count + 2)
                decode_targets.append(torch.tensor(sequence, device=operations.device))
                decode_states.append(self._start(meaning))
        losses.append(len(examples) * F.cross_entropy(operations, operations.new_tensor(targets, dtype=torch.long)))
        if decode_targets:
            labels = nn.utils.rnn.pad_sequence(decode_targets, batch_first=True, padding_value=-100)
            embeddings = self._embeddings()
            teacher = F.embedding(labels[:, :-1].clamp_min(0), embeddings)
            inputs = torch.cat((teacher.new_zeros(len(labels), 1, self.width), teacher), dim=1)
            decoded, _ = self.decoder(inputs, torch.stack(decode_states)[None])
            losses.append(len(examples) * F.cross_entropy(self.word_head(decoded).flatten(0, 1), labels.flatten()))
        return torch.stack(losses).sum() / max(1, len(examples))

    def compose(self, program, registry):
        if not 0 < len(program.leaves) <= 128:
            return None
        operations, pointers, modes = self.encode((program,))
        probabilities = operations.softmax(-1)[0]
        choice = int(probabilities.detach().argmax())
        if choice == len(self.operation_ids) or float(probabilities[choice].detach()) < .7:
            return None
        semantic_id = self.operation_ids[choice]
        try:
            operation = registry.operation_spec(semantic_id)
            descriptor = registry.descriptors[semantic_id]
            vp = registry._reference((descriptor.domain, semantic_id))
            vp_value = registry._payload(vp)
        except (ValueError, RuntimeError, KeyError):
            return None
        if tuple(operation.operand_roles) != ('I1', 'I2'):
            return None
        indices = pointers[0].detach().argmax(-1).tolist()
        refs = tuple(('sym', int(program.concept_ids[index])) for index in indices)
        if any(reference[1] <= 0 for reference in refs):
            return None
        return ConceptualMeaning(torch.stack((program.leaves[indices[0]], vp_value,
                                              program.leaves[indices[1]])),
            torch.ones(3, dtype=torch.bool, device=vp_value.device),
            mode='interrogative' if int(modes.detach().argmax(-1)[0]) else 'assertive',
            role_refs=(refs[0], vp, refs[1]))

    def generate(self, meaning, registry, *, max_words):
        if not 0 < max_words <= 128:
            raise ValueError("linguistic generation requires a 1–128 word budget")
        try:
            signature = registry.signature_for(ConceptualMeaning(
                meaning.roles, meaning.role_mask, **dict(meaning.metadata(), mode='interrogative')))
        except (RuntimeError, ValueError):
            return None
        if (signature.operation.semantic_id not in self.operation_ids
                or tuple(signature.occupied_roles) != ('I1', 'I2') or meaning.constituents):
            return None
        embeddings = self._embeddings()
        state = self._start(meaning)[None, None]
        current = embeddings.new_zeros(1, 1, self.width)
        words, selections = [], []
        count = len(self.word_rows)
        for _ in range(max_words + 1):
            decoded, state = self.decoder(current, state)
            choice = int(self.word_head(decoded[0, 0]).detach().argmax())
            if choice == count + 2:
                return GeneratedMeaning(torch.stack(words) if words else embeddings[:0], tuple(selections), False)
            if len(words) == max_words:
                break
            if choice < count:
                words.append(self.word_values[choice])
                selections.append(('word', int(self.word_rows[choice])))
            else:
                slot = (0, 2)[choice - count]
                words.append(meaning.roles[slot])
                selections.append(('role', slot))
            current = embeddings[choice][None, None]
        return GeneratedMeaning(torch.stack(words), tuple(selections), True)
