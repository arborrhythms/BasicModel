def _reconstructionReport(self):
    """Run a final eval pass and print input reconstruction per row."""
    data = self.inputSpace.data
    split = "test"
    split_input = getattr(data, f"{split}_input", None)
    split_output = getattr(data, f"{split}_output", None)
    if self._data_len(split_input) == 0:
        split = "train"
        split_input = getattr(data, f"{split}_input", None)
        split_output = getattr(data, f"{split}_output", None)
    n_rows = self._data_len(split_input)
    if n_rows == 0:
        return
    try:
        max_rows = int(os.environ.get("BASIC_RECON_REPORT_MAX", "64"))
    except ValueError:
        max_rows = 64
    if max_rows > 0 and n_rows > max_rows:
        TheMessage(f"=== Input Reconstruction ({split}) ===")
        TheMessage(
            f"  skipped: {n_rows} rows exceeds "
            f"BASIC_RECON_REPORT_MAX={max_rows}; set it to 0 to "
            f"print every row.")
        return

    self.set_sigma(0)
    try:
        _, _, allOut, allIn = self.runEpoch(batchSize=n_rows, split=split)
    finally:
        self.set_sigma(0.5)

    if isinstance(allOut, torch.Tensor):
        data.reconstructed_output = [x.detach().cpu() for x in allOut]
    _objective_probe.log("endpoint_predictions_saved", split=split, rows=n_rows, console_rendering=False)
    return

    if not isinstance(allIn, torch.Tensor) or allIn.numel() == 0:
        TheMessage(f"=== Input Reconstruction ({split}) ===")
        TheMessage("  (no reconstructed input was produced)")
        return

    n = min(n_rows, int(allIn.shape[0]))
    originals = self._slice_data(split_input, n)
    labels = self._slice_data(split_output, n)
    reconstructed = self._decode_reconstructed_inputs(allIn[:n], originals)

    rows = []
    TheMessage(f"=== Input Reconstruction ({split}) ===")
    for i in range(n):
        original = self._display_value(originals[i])
        recon = reconstructed[i] if i < len(reconstructed) else ""
        label = self._display_value(labels[i]) if i < len(labels) else ""
        pred = ""
        if isinstance(allOut, torch.Tensor) and allOut.numel() > 0:
            pred = self._display_value(allOut[i])
        orig_words = original.replace("\x00", " ").split()
        recon_words = recon.replace("\x00", " ").split()
        match = orig_words == recon_words
        status = "OK" if match else "MISMATCH"
        css = "match" if match else "mismatch"
        TheMessage(
            f"  row[{i}] input={original!r} -> "
            f"reconstructed={recon!r} label={label} "
            f"predicted={pred} {status}")
        rows.append([
            original,
            f'<span class="{css}">{recon}</span>',
            label,
            pred,
            f'<span class="{css}">{"Yes" if match else "No"}</span>',
        ])

    TheReport.add_table(
        f"Input vs Reconstructed ({split})",
        ["Input", "Reconstructed", "Label", "Predicted", "Match"],
        rows)
    self.inputSpace.data.reconstructed_input = reconstructed
    if isinstance(allOut, torch.Tensor):
        self.inputSpace.data.reconstructed_output = [
            allOut[i].detach().cpu() for i in range(min(n, allOut.shape[0]))]
