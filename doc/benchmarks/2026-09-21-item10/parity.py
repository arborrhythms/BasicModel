"""Native reconstruction measurement, preserving row and sentence order.

Both modes start in fresh processes at the same recorded seed and config.
The four validation presentations are two consecutive sentences per row.
There are no optimizer steps or learned-utility assertions in this probe.
"""
import hashlib
import torch
from What import What


def _fingerprint(values):
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def measure(model, mode):
    model._install_unit_span_fn()  # runEpoch installs the unit lexer before packing
    inputs = model.inputSpace.data.validation_input
    surfaces = [model._bytes_to_text(value).rstrip("\x00") for value in inputs[:4]]
    if len(surfaces) != 4:
        raise RuntimeError("parity workload requires four validation sentences")
    # Allocate identical word/object rows before changing batch layout. Fresh
    # on-demand lexical allocation is order dependent; comparing those different
    # identities would confound the reconstruction parity measurement.
    raw = model.inputSpace.prepInput(surfaces)
    with torch.no_grad():
        model.runBatch(train=False, batchSize=4, split="validation",
                       batch_override=(raw, torch.empty(4, 0)), source_rows=list(range(4)),
                       questions=tuple(What.present(i, split="validation") for i in range(4)))
    model.flush_word_buffers()
    model.dispatch_per_row_reset([True] * 4)
    model.dispatch_soft_reset()
    model.post_tick_compact()
    rows = [surfaces[:2], surfaces[2:]]
    addresses = [[0, 1], [2, 3]]
    report = {"mode": mode, "rows": rows, "source_rows": addresses,
              "initial_parameters_sha256": _fingerprint(model.named_parameters()),
              "initial_dictionary_sha256": _fingerprint([("W", model.conceptualSpace.similarity_codebook.W)]),
              "vocabulary_warmup": surfaces, "steps": [], "sentences": []}
    # The packed lexer assigns the joining space to the preceding sentence.
    # Feed that same space to the corresponding single presentation, so this
    # comparison has identical byte/unit streams, including whitespace units.
    report["boundary_convention"] = "joining space belongs to the first sentence in each row"
    batches = [(rows, addresses)] if mode == "packed" else [
        ([row[i] + (" " if i == 0 else "") for row in rows], [row[i] for row in addresses]) for i in range(2)]
    for index, (batch, source) in enumerate(batches):
        raw = (model.inputSpace.prepPackedInput(batch) if mode == "packed"
               else model.inputSpace.prepInput(batch))
        teacher = getattr(model, "teacher", None)
        if teacher is not None:
            teacher.stage_batch_sources("validation", source)
        with torch.no_grad():
            result, _ = model.runBatch(
                train=False, batchNum=index, batchSize=2, split="validation",
                batch_override=(raw, torch.empty(2, 0)), source_rows=source,
                questions=tuple(What.present(row, split="validation") for row in range(2)))
        active = model.inputSpace._word_active_mask
        ids = (model.inputSpace._packed_sentence_ids if mode == "packed"
               else torch.zeros_like(active, dtype=torch.long))
        report["steps"].append({"lossIn": float(result.lossIn),
                                "truncated": model._recon_truncated.tolist()})
        costs = model._recon_sentence_costs
        roots = model._tensor_sentence_roots_live
        reference = model._tensor_pushed_ideas
        recovered = model._recon_ideas
        dictionary = model.conceptualSpace.similarity_codebook.W
        for row in range(2):
            for sentence in range(2 if mode == "packed" else 1):
                address = source[row][sentence] if mode == "packed" else source[row]
                mask = active[row] & (ids[row] == sentence)
                report["sentences"].append({
                    "source_row": address, "text": surfaces[address],
                    "word_count": int(mask.sum()),
                    "byte_cost": float(costs[row, sentence]),
                    "root": (model._last_understanding.answer_program[row].end_state.reshape(-1)
                             if sentence == (1 if mode == "packed" else 0)
                             else roots[row, sentence]).detach().tolist(),
                    "reference": reference[row, mask].detach().tolist(),
                    "recovered": recovered[row, mask].detach().tolist(),
                    "dictionary_sha256": _fingerprint([("W", dictionary)]),
                })
        model.flush_word_buffers()
        hard_eos = [index == len(batches) - 1] * 2
        if mode == "packed":
            model.dispatch_packed_soft_reset(hard_eos)
        model.dispatch_per_row_reset(hard_eos)
        model.dispatch_soft_reset()
        model.post_tick_compact()
    report["sentences"].sort(key=lambda row: row["source_row"])
    report["mean_sentence_byte_cost"] = sum(row["byte_cost"] for row in report["sentences"]) / 4
    return report
