"""Small feature variants of configurations retained by the October 2 review."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def parallel_concepts(*, category=False):
    """Keep the mixing path, with the tested sparse order and optional category.

    The former sparse/masked XML files differed in these feature settings.
    Geometry is bounded here; the MM_20M XOR proofs still use their full XML.
    """
    source = (ROOT/'data/MM_20M_xor.xml').read_text()
    source = source.replace('1024', '64').replace('65536', '512').replace('8192', '512')
    tree = ET.fromstring(source)
    for key, value in {
        'architecture/symbolicOrder': '3',
        'architecture/categoryCodebook': str(category).lower(),
        'ConceptualSpace/nVectors': '64' if category else '32',
        'WholeSpace/codebook': 'quantize',
        'architecture/training/conceptualSimilarityScale': '0.0',
        'architecture/training/reconstructionScale': '1.0' if category else '0.1',
        'architecture/training/maskRate': '0.3',
    }.items():
        parent, tag = key.rsplit('/', 1)
        section = tree.find(parent)
        element = section.find(tag)
        if element is None:
            element = ET.SubElement(section, tag)
        element.text = value
    with tempfile.TemporaryDirectory(prefix='parallel-concepts-') as directory:
        path = Path(directory)/'MM_20M_xor.xml'
        ET.ElementTree(tree).write(path, encoding='unicode')
        yield str(path)


@contextmanager
def small_retained(name):
    """Same retained reading and feature switches at smoke-test geometry."""
    tree = ET.parse(ROOT/'data'/name)
    for node in tree.getroot().iter():
        if node.tag in ('nDim', 'nInputDim', 'nOutputDim') and node.text and int(node.text) >= 64:
            node.text = '16'
        elif node.tag in ('nVectors', 'activeVectors') and node.text and int(node.text) >= 512:
            node.text = '512'
        elif node.tag in ('nInput', 'nOutput') and node.text and int(node.text) >= 256:
            node.text = '256'
    with tempfile.TemporaryDirectory(prefix='retained-smoke-') as directory:
        path = Path(directory)/Path(name).name
        tree.write(path, encoding='unicode')
        yield str(path)


def freeze_admission(model, inputs, monkeypatch):
    """Keep a repeated-read comparison on one admitted percept dictionary."""
    import torch
    from Spaces import Space
    with torch.no_grad():
        model.forward(inputs)
    for space in model.modules():
        if isinstance(space, Space):
            monkeypatch.setattr(space, '_online_learning_frozen', True, raising=False)
