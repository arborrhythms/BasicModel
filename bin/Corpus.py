"""Versioned corpus inputs and document-preserving curriculum streams.

Age and stage metadata schedule presentations; neither is supplied to the
model as a lexical class, order, answer or source of truth.
"""
import hashlib
import json
from pathlib import Path

from data import SentenceStreamDataset


SPLITS = ('train', 'validation', 'test')


def admit_training_forms(model, texts, answers=(), *, split='train'):
    """Admit the phase's training vocabulary, without presenting its meanings.

    This eager phase boundary uses the same native atoms and word rows as
    reading. It cannot mint an object, open an occurrence or prime a word.
    """
    if split != 'train':
        raise ValueError('form admission accepts training data only')
    from Meronomy import percept_spans
    owner=model._concept_owner()
    identity=model.perceptualSpace.percept_store.identity
    forms=set()
    from itertools import chain
    for text in chain(texts,(answer for answer in answers if isinstance(answer,str))):
        raw=text.encode('ascii',errors='replace')
        forms.update(raw[start:end] for start,end in percept_spans(raw) if end>start)
    admitted=[]
    for raw in sorted(forms):
        parts=identity.admit(raw)
        word=owner.interpret.lookup_word(parts,(),form=raw,word_reading=False)
        if word is None:
            raise RuntimeError(f'training form admission failed: {raw!r}')
        admitted.append((raw,word))
    return tuple(admitted)


def admit_stage(model, dataset, source_rows):
    """The training stage boundary precedes native input construction."""
    if not isinstance(dataset, CurriculumStreamDataset):
        return
    rows = [i for stream in source_rows for i in stream]
    stages = {index for index,(lo,hi,_) in enumerate(dataset._stage_windows)
              if any(lo <= i < hi for i in rows)}
    if len(stages) != 1:
        raise ValueError('a corpus brick cannot mix curriculum stages')
    stage = next(iter(stages))
    key = (dataset.source_token, stage)
    done = model.__dict__.setdefault('_corpus_vocab_stages',set())
    if key in done:
        return
    lo,hi,_ = dataset._stage_windows[stage]
    indices = range(lo,hi)
    answers = (model.inputSpace.data.text_answers or {}).get('train',())
    admitted = admit_training_forms(model,(dataset.inputs[i] for i in indices),
        (answers[i] for i in indices if i < len(answers)))
    done.add(key)
    model._last_corpus_admission = dict(stage=stage,forms=len(admitted),sentences=len(indices))


def unit_capacity_bound(text):
    """Pinned words plus every other byte, including the packing separator.

    The native tiler cannot split maximal letter words. A remaining byte
    can occupy at most one unit. This bound therefore survives later learned
    punctuation/whitespace cuts; it does not freeze the initial tiling.
    """
    from Meronomy import word_spans
    raw = text.encode('ascii', errors='replace') + b' '
    words = word_spans(raw)
    return len(words) + len(raw) - sum(end - start for start, end in words)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def corpus_rows(path):
    """Verify every shard before yielding its exact, independently split rows."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('version') != 1 or not manifest.get('shards'):
        raise ValueError('curriculum requires a version-1 manifest with shards')
    files, seen_paths = [], set()
    for shard in manifest['shards']:
        candidate = (path.parent / shard['path']).resolve()
        if not candidate.is_relative_to(path.parent) or candidate in seen_paths:
            raise ValueError('curriculum shard paths must be distinct and inside the manifest directory')
        seen_paths.add(candidate)
        if file_digest(candidate) != shard['sha256']:
            raise ValueError(f'curriculum shard hash differs: {shard["path"]}')
        if type(shard['rows']) is not int or shard['rows'] <= 0:
            raise ValueError('curriculum shard row count must be positive')
        files.append((candidate, shard))
    documents, last_stage = {}, {split: -1 for split in SPLITS}
    current_document = {split: None for split in SPLITS}
    last_sentence = {}
    for candidate, shard in files:
        count = 0
        with candidate.open() as source:
            for line in source:
                row = json.loads(line)
                split, stage = row['split'], row['stage']
                document, sentence = row['document'], row['sentence']
                if (split not in SPLITS or type(stage) is not int or stage < 0
                        or not isinstance(document, str) or not document
                        or type(sentence) is not int or sentence < 0
                        or not isinstance(row['text'], str) or not row['text'].strip()):
                    raise ValueError('invalid curriculum split, stage, document, sentence or text')
                if stage < last_stage[split]:
                    raise ValueError('curriculum stages must be chronological within each split')
                if document in documents:
                    if documents[document] != (split, stage):
                        raise ValueError('a curriculum document cannot cross splits or stages')
                    if current_document[split] != document:
                        raise ValueError('a curriculum document must be contiguous')
                    if sentence <= last_sentence[document]:
                        raise ValueError('curriculum sentence addresses must increase within a document')
                documents[document] = (split, stage)
                current_document[split] = document
                last_sentence[document] = sentence
                last_stage[split] = stage
                count += 1
                yield row
        if count != shard['rows']:
            raise ValueError(f'curriculum shard row count differs: {shard["path"]}')


def load_curriculum(data, config, *, max_docs=None, max_sentence_words=None):
    """Load immutable whole sentences; retain stage and source addresses."""
    from util import ProjectPaths
    import Meronomy
    name = config.get('corpusManifest')
    if not name:
        raise ValueError('curriculum dataset requires corpusManifest')
    path = Path(name)
    if not path.is_absolute():
        path = Path(ProjectPaths.PROJECT_DIR) / path
    manifest = json.loads(path.read_text())
    texts = {split: [] for split in SPLITS}
    addresses = {split: [] for split in SPLITS}
    admitted, excluded, documents = set(), 0, {}
    excluded_units = 0
    max_units = config.get('maxSentenceUnits')
    if max_units is not None and (type(max_units) is not int or max_units < 1):
        raise ValueError('maxSentenceUnits must be a positive integer')
    for row in corpus_rows(path):
        document = row['document']
        if document not in documents:
            documents[document] = len(documents)
            if max_docs is None or len(admitted) < int(max_docs):
                admitted.add(document)
        if document not in admitted:
            continue
        text, split = row['text'], row['split']
        if max_sentence_words is not None and len(Meronomy.word_spans(
                text.encode('ascii', errors='replace'))) > int(max_sentence_words):
            excluded += 1
            continue
        if max_units is not None and unit_capacity_bound(text) > max_units:
            excluded_units += 1
            continue
        index = len(texts[split])
        texts[split].append(text)
        addresses[split].append(dict(split=split, row=index, document=documents[document],
            sentence=row['sentence'], stage=row['stage'], char_start=0, char_end=len(text),
            external_id=row.get('external_id', document), source_time=None))
    if not texts['train']:
        raise ValueError('curriculum has no admitted training sentences')
    data.source_manifest = dict(dataset='curriculum', version=1, corpus=manifest.get('corpus'),
        manifest_sha256=file_digest(path), shards=manifest['shards'],
        sources=manifest.get('sources', []), stages=manifest.get('stages', []),
        max_docs=max_docs, max_sentence_words=max_sentence_words,
        excluded_overlong=excluded, admitted_documents=len(admitted),
        max_sentence_units=max_units, excluded_unit_bound=excluded_units,
        stream_policy='stage_barriers_contiguous_documents_v1')
    data.processLM({split: dict(text=texts[split], label=[]) for split in SPLITS})
    data.source_addresses = addresses


class CurriculumStreamDataset(SentenceStreamDataset):
    """Finish a stage in every stream before advancing to the next stage.

    Whole documents are partitioned into contiguous stream windows within
    each stage. Uneven tails and stages smaller than the stream count are
    retained, using the packed cursor's existing empty-row masks.
    """
    def __init__(self, inputs, num_streams, *, stages, document_ids,
                 outputs=None, slab_bytes=None):
        if slab_bytes is not None:
            raise ValueError('curriculum requires complete-sentence packing')
        super().__init__(inputs, num_streams, outputs=outputs, document_ids=document_ids)
        if len(stages) != len(inputs) or document_ids is None:
            raise ValueError('curriculum needs stage and document addresses for every sentence')
        if any(type(stage) is not int or stage < 0 for stage in stages):
            raise ValueError('curriculum stages must be nonnegative integers')
        if any(a > b for a, b in zip(stages, stages[1:])):
            raise ValueError('curriculum stages must be chronological')
        self._stage_windows = []
        start, previous = 0, None
        seen = set()
        for index in range(len(inputs) + 1):
            if index < len(inputs):
                document = document_ids[index]
                if document != previous:
                    if document in seen:
                        raise ValueError('curriculum documents must be contiguous')
                    seen.add(document)
                elif index > 0 and stages[index] != stages[index - 1]:
                    raise ValueError('a curriculum document cannot cross stages')
                previous = document
            if index == len(inputs) or stages[index] != stages[start]:
                boundaries = [start] + [i for i in range(start + 1, index)
                    if document_ids[i] != document_ids[i - 1]] + [index]
                count = len(boundaries) - 1
                windows = [(boundaries[b * count // num_streams],
                            boundaries[(b + 1) * count // num_streams])
                           for b in range(num_streams)]
                self._stage_windows.append((start, index, windows))
                start = index
        self._stage_index = 0
        self.source_token = None
        self._layout_token = hashlib.sha256(json.dumps(self._stage_windows,
            separators=(',', ':')).encode()).hexdigest()

    def cursor_state(self):
        """Consumed positions, independent of future changes to the tiler."""
        return dict(version=1, streams=self.num_streams, sentences=len(self.inputs),
                    source=self.source_token, layout=self._layout_token,
                    stage=self._stage_index,
                    steps=None if self._packed_steps is None else list(self._packed_steps))

    def restore_cursor(self, state):
        fixed = self.cursor_state()
        if any(state.get(key) != fixed[key] for key in
               ('version', 'streams', 'sentences', 'source', 'layout')):
            raise ValueError('curriculum resume source or stream layout differs')
        stage, steps = state.get('stage'), state.get('steps')
        if type(stage) is not int or not 0 <= stage < len(self._stage_windows):
            raise ValueError('invalid curriculum resume stage')
        if steps is not None and (not isinstance(steps, list) or len(steps) != self.num_streams
                or any(type(step) is not int or not 0 <= step <= end - start
                       for step, (start, end) in zip(steps, self._stage_windows[stage][2]))):
            raise ValueError('invalid curriculum resume positions')
        self._stage_index, self._packed_steps = stage, None if steps is None else list(steps)
        self.last_source_indices = None

    def _packed_window(self, stream):
        return self._stage_windows[self._stage_index][2][stream]

    def _packed_boundary(self, previous, current):
        return self.document_ids[previous] != self.document_ids[current]

    def _stage_done(self):
        return self._packed_steps is not None and all(
            step == end - start for step, (start, end) in
            zip(self._packed_steps, self._stage_windows[self._stage_index][2]))

    def next_packed_tick(self, *args, **kwargs):
        if self._stage_done() and self._stage_index + 1 < len(self._stage_windows):
            self._stage_index += 1
            self._packed_steps = None
        return super().next_packed_tick(*args, **kwargs)

    def packed_done(self):
        return self._stage_index + 1 == len(self._stage_windows) and self._stage_done()

    def all_done(self):
        return self.packed_done()

    def next_tick(self):
        raise ValueError('curriculum requires packSentences=true')

    def __iter__(self):
        raise ValueError('curriculum requires the packed sentence cursor')

    def reset_cursor(self):
        super().reset_cursor()
        self._stage_index = 0

    def progress(self):
        consumed = self._stage_windows[self._stage_index][0] + sum(self._packed_steps or ())
        return consumed / len(self.inputs)
