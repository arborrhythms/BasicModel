"""data/model.xml is the one defaults file (sigma-pi plan 19.123): every key the
code reads from TheXMLConfig by a literal name has its default there."""
import ast
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SPACES = ('InputSpace', 'PartSpace', 'ModalSpace', 'ConceptualSpace',
          'WholeSpace', 'OutputSpace', 'SymbolSpace')

# A read of these refuses a retired key; a default would refuse every model.
RETIRED = {'architecture.prediction', 'architecture.sparseReplace',
           'architecture.whatThinkingPrimitives', 'SymbolSpace.language.useGrammar',
           'architecture.training.answerLossWeight', 'architecture.training.predictNextLossWeight',
           'conceptIds', 'lexer', 'chunking'}
# Absent, these are made from another key or from the codebook as built.
DERIVED = {'architecture.sentenceProtocol', 'architecture.reasoningIterations',
           'architecture.truthSet', 'commitmentDecay', 'codebookEmaDeadThreshold'}


def defaults():
    paths = set()
    def walk(node, path):
        paths.add(path)
        for name in node.attrib:
            paths.add(path + '.' + name)
        for child in node:
            walk(child, path + '.' + child.tag if path else child.tag)
    walk(ET.parse(ROOT/'data'/'model.xml').getroot(), '')
    return paths


def literal_reads():
    """(kind, section or None, dotted key, place) of every literal read in bin/."""
    for path in sorted((ROOT/'bin').glob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            owner = node.func.value
            if not ((isinstance(owner, ast.Name) and owner.id == 'TheXMLConfig') or
                    (isinstance(owner, ast.Attribute) and owner.attr == 'TheXMLConfig')):
                continue
            kind, args, place = node.func.attr, node.args, f'{path.name}:{node.lineno}'
            if kind == 'space' and len(args) > 1 and isinstance(args[1], ast.Constant):
                section = args[0].value if isinstance(args[0], ast.Constant) else None
                yield 'space', section, args[1].value, place
            elif kind in ('get', 'training', 'data_param') and args and isinstance(args[0], ast.Constant):
                prefix = {'get': '', 'training': 'architecture.training.',
                          'data_param': 'architecture.data.'}[kind]
                yield 'get', None, prefix + args[0].value, place


def test_every_literal_read_has_its_default_in_model_xml():
    known = defaults()
    missing = []
    for kind, section, key, place in literal_reads():
        if key in RETIRED or key in DERIVED:
            continue
        if kind == 'get':
            found = key in known
        elif section is not None:
            found = f'{section}.{key}' in known or f'architecture.{key}' in known
        else:
            found = (any(f'{name}.{key}' in known for name in SPACES)
                     or f'architecture.{key}' in known)
        if not found:
            missing.append(f'{place}: {section or ""} {key}')
    assert not missing, 'keys read with no default in data/model.xml:\n' + '\n'.join(missing)


def test_the_exceptions_are_still_read():
    """An exception that nothing reads any more is removed from the lists."""
    read = {key.split('.')[-1] if kind == 'space' else key for kind, _, key, _ in literal_reads()}
    read |= {key for kind, _, key, _ in literal_reads()}
    assert not (RETIRED | DERIVED) - read
