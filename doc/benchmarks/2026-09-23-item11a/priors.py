"""Compare learned prior memberships and ladder cuts with the baseline tags."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
import torch
import Spaces
from bounded_tests import source_snapshot
from test_wholespace_property_migration import _small_property_model


def cuts(raw, lookup, discarded, digit_bits):
    spans = []
    i = 0
    while i < len(raw):
        signature = int(lookup[raw[i]])
        j = i + 1
        while j < len(raw) and int(lookup[raw[j]]) == signature and not signature & digit_bits:
            j += 1
        if signature and not signature & discarded:
            spans.append((i, j, signature))
        i = j
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = source_snapshot(ROOT)
    baseline = subprocess.check_output(['git', 'show', '17403a1:bin/Spaces.py'], cwd=ROOT, text=True)
    builder = next(n for n in ast.parse(baseline).body
                   if isinstance(n, ast.FunctionDef) and n.name == '_build_property_signature_lut')
    namespace = dict(vars(Spaces))
    exec(compile(ast.Module(body=[builder], type_ignores=[]), '<baseline-prior-builder>', 'exec'), namespace)
    old_lut, old_discard = namespace['_build_property_signature_lut']()
    with tempfile.TemporaryDirectory() as directory:
        model = _small_property_model(Path(directory))
        ws = model.wholeSpace
        primitive = ws.subspace.what.primitive_properties
        current, discard = Spaces._analysis_property_signature(ws)
        target = torch.zeros_like(primitive.members)
        for row, (_, kind) in enumerate(Spaces._CANONICAL_PROPERTY_ROWS):
            for lo, hi in Spaces._CHAR_CLASS_RANGES[kind]:
                target[row, lo:hi + 1] = 1
        definitions = primitive(torch.arange(256)).t().detach()
        corpus = ROOT / 'data/MM_ladder_idiom.xml'
        sentences = [s for node in ET.parse(corpus).getroot().findall('architecture/data/input')
                     for s in (node.text or '').split('|')]
        digit_bits = Spaces._digit_signature_bits(ws)
        comparisons = []
        for text in sentences:
            raw = text.encode('latin1', 'replace')
            old = cuts(raw, old_lut, old_discard, digit_bits)
            new = cuts(raw, current, discard, digit_bits)
            comparisons.append(dict(text=text, before=old, after=new, equal=old == new))
        report = dict(baseline='17403a1', source=source,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      memberships=list(definitions.shape),
                      max_prior_error=float((definitions-target).abs().max()),
                      differing_byte_signatures=int((old_lut != current).sum()),
                      discarded_equal=old_discard == discard,
                      corpus=str(corpus.relative_to(ROOT)),
                      corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
                      sentences=len(sentences), differing_sentences=sum(not c['equal'] for c in comparisons),
                      runs=sum(len(c['after']) for c in comparisons), comparisons=comparisons,
                      source_unchanged=source_snapshot(ROOT) == source)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('source', 'comparisons')}, indent=2))
        if report['max_prior_error'] or report['differing_byte_signatures'] or report['differing_sentences']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
