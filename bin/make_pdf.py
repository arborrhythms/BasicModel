"""Build one PDF from the Markdown chapters: Markdown to Typst, then Typst to PDF.

The conversion is md2typst, the math is converted from LaTeX by tylax, and
the typst compiler writes the PDF. None of them needs pandoc or a TeX
installation. A formula that does not convert stops the build with its text.
"""
import argparse
import json
import re
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import md2typst
import tylax
import typst
from md2typst.image_attrs import fold_image_attrs

MATH = tylax.L2TOptions(frac_to_slash=False)
ATTRIBUTE = re.compile(r'\s*\{#([A-Za-z0-9_:.-]+)\}\s*$')


def _open_commas(group):
    """Replace the commas that stand outside every bracket of a fraction's argument."""
    out, depth, i = [], 0, 0
    while i < len(group):
        ch = group[i]
        if ch == '\\':
            out.append(group[i:i + 2])
            i += 2
            continue
        depth += ch in '([{'
        depth -= ch in ')]}'
        out.append(r'\text{,}\,' if ch == ',' and depth == 0 else ch)
        i += 1
    return ''.join(out)


def _fraction_commas(body):
    """Typst reads a bare comma inside frac(...) as a separator of its arguments."""
    out, i = [], 0
    for match in re.finditer(r'\\[dt]?frac(?![A-Za-z])', body):
        if match.start() < i:
            continue
        j, groups = match.end(), []
        for _ in range(2):
            while j < len(body) and body[j].isspace():
                j += 1
            if j >= len(body) or body[j] != '{':
                groups = None
                break
            depth, k = 0, j
            while k < len(body):
                if body[k] == '\\':
                    k += 2
                    continue
                depth += body[k] == '{'
                depth -= body[k] == '}'
                if depth == 0:
                    break
                k += 1
            groups.append(body[j + 1:k])
            j = k + 1
        if groups is None:
            continue
        out.append(body[i:match.start()] + match.group(0)
                   + ''.join('{' + _open_commas(_fraction_commas(group)) + '}' for group in groups))
        i = j
    return ''.join(out) + body[i:]


def math(latex):
    """One LaTeX formula as Typst math, without its delimiters."""
    # tylax writes \mathbb{RP} as one unknown name; each letter is its own symbol.
    latex = re.sub(r'\\mathbb\{([A-Za-z]{2,})\}',
                   lambda m: ''.join(r'\mathbb{%s}' % letter for letter in m.group(1)), latex)
    return tylax.latex_to_typst(_fraction_commas(latex), options=MATH).strip().strip('$').strip()


def plain(node):
    """The text of a node without its markup."""
    content = getattr(node, 'content', None)
    if isinstance(content, str):
        return content
    return ''.join(plain(child) for child in getattr(node, 'children', ()) or ())


def slug(text):
    """The anchor GitHub gives a heading, kept to what a Typst label may hold."""
    text = ATTRIBUTE.sub('', text).strip().lower()
    return re.sub(r'[^a-z0-9_-]', '', text.replace(' ', '-'))


def walk(node):
    yield node
    if is_dataclass(node):
        for field in fields(node):
            value = getattr(node, field.name)
            for child in value if isinstance(value, (tuple, list)) else (value,):
                if is_dataclass(child):
                    yield from walk(child)


class Chapter:
    """One Markdown file, its parsed form and the anchors of its headings."""

    def __init__(self, path, root):
        self.path = Path(path)
        self.key = re.sub(r'[^a-z0-9]', '', self.path.stem.lower())
        self.folder = self.path.resolve().parent.relative_to(root)
        self.document = fold_image_attrs(md2typst.get_parser(None).parse(self.path.read_text(encoding='utf-8')))
        self.anchors, self.labels, seen = {}, {}, {}
        for node in walk(self.document):
            if type(node).__name__ != 'Heading':
                continue
            text = plain(node)
            base = slug(text)
            count = seen.get(base, 0)
            seen[base] = count + 1
            anchor = base if count == 0 else f'{base}-{count}'
            label = f'{self.key}:{anchor}' if anchor else None
            self.labels[id(node)] = label
            if label:
                self.anchors[anchor] = label
                explicit = ATTRIBUTE.search(text)
                if explicit:
                    self.anchors[explicit.group(1)] = label
        self.top = next((label for label in self.labels.values() if label), None)


class Generator(md2typst.TypstGenerator):
    """md2typst's writer, with the document's own links, anchors, breaks and math."""

    def __init__(self, chapter, chapters, root):
        super().__init__()
        self.chapter, self.chapters, self.root = chapter, chapters, root

    def visit_SoftBreak(self, node):
        # A wrapped source line is not a line break of the document, and a
        # new line that begins with "/", "-", "+" or "=" would be read as markup.
        return ' '

    def visit_Heading(self, node):
        text = re.sub(r'\s*\{\\#[A-Za-z0-9_:.-]+\}\s*$', '', super().visit_Heading(node))
        label = self.chapter.labels.get(id(node))
        return f'{text} <{label}>' if label else text

    def visit_Link(self, node):
        url = node.url
        if re.match(r'[a-z][a-z0-9+.-]*:', url):
            return super().visit_Link(node)
        content = self._visit_children_inline(node.children)
        target, _, anchor = url.partition('#')
        chapter = self.chapter
        if target:
            resolved = (self.root / self.chapter.folder / target).resolve()
            chapter = next((c for c in self.chapters if c.path.resolve() == resolved), None)
        label = None
        if chapter is not None:
            label = chapter.anchors.get(anchor.lower()) if anchor else chapter.top
        # A link to a file outside the document keeps its text and loses its target.
        return f'#link(<{label}>)[{content}]' if label else content

    def visit_Image(self, node):
        text = super().visit_Image(node)
        if re.match(r'[a-z][a-z0-9+.-]*:|/', node.url):
            return text
        # typst reads a path that starts with "/" from the project root.
        return text.replace(f'"{node.url}"', '"/' + (self.chapter.folder / node.url).as_posix() + '"', 1)

    def visit_HtmlInline(self, node):
        return self._html(node, super().visit_HtmlInline(node))

    def visit_HtmlBlock(self, node):
        return self._html(node, super().visit_HtmlBlock(node))

    def _html(self, node, text):
        source = plain(node) or text
        if re.fullmatch(r'\s*<br\s*/?>\s*', source):
            return ' \\\n'
        if re.fullmatch(r'\s*(<a\s+(id|name)="[^"]*">\s*)?(</a>)?\s*', source):
            return ''
        return text


CALL = re.compile(r'#(mi|mitex)\("((?:[^"\\]|\\.)*)"\)', re.S)
RAW = re.compile(r'#(mi|mitex)\(`([^`]*)`\)', re.S)


def formulas(text):
    """Replace md2typst's calls of its LaTeX package by Typst math."""
    found = []

    def native(match):
        body = match.group(2)
        if match.re is CALL:
            body = json.loads('"' + body.replace('\n', '\\n') + '"')
        found.append(body)
        converted = math(body)
        return f'$ {converted} $' if match.group(1) == 'mitex' else f'${converted}$'

    text = RAW.sub(native, CALL.sub(native, text))
    return re.sub(r'#import "@preview/mitex:[^"]*": \*\n', '', text), found


HEAD = '''#set document(title: "{title}")
#set page(paper: "us-letter", margin: 1in, numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#show raw.where(block: true): set text(size: 8.5pt)
#show table: set text(size: 9pt)
#show link: set text(fill: rgb("#1a4f8b"))
#align(center + horizon, text(26pt, weight: "bold")[{title}])
#pagebreak()
#outline(depth: 2)
'''


def failing_formula(found, scratch):
    """Name the first formula the compiler rejects, for the error message."""
    probe = scratch / 'formula.typ'
    for body in dict.fromkeys(found):
        probe.write_text(f'$ {math(body)} $', encoding='utf-8')
        try:
            typst.compile(str(probe), output=str(scratch / 'formula.pdf'))
        except typst.TypstError as error:
            return f'{error}\n  LaTeX: {body}\n  Typst: {math(body)}'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('chapters', nargs='+', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--title', default='Basic Model')
    parser.add_argument('--work', type=Path, default=Path('output/pdf'),
                        help='where the intermediate Typst file is written, inside the project')
    args = parser.parse_args()
    root = Path.cwd().resolve()
    chapters = [Chapter(path, root) for path in args.chapters]
    parts, found = [], []
    for chapter in chapters:
        text, bodies = formulas(Generator(chapter, chapters, root).generate(chapter.document))
        found.extend(bodies)
        parts.append(f'// {chapter.path}\n#pagebreak(weak: true)\n{text}')
    imports = sorted(set(re.findall(r'^#import "@preview/[^\n]*\n', ''.join(parts), flags=re.M)))
    body = re.sub(r'^#import "@preview/[^\n]*\n', '', '\n\n'.join(parts), flags=re.M)
    args.work.mkdir(parents=True, exist_ok=True)
    source = args.work / (args.output.stem + '.typ')
    source.write_text(''.join(imports) + HEAD.format(title=args.title) + body, encoding='utf-8')
    try:
        typst.compile(str(source), output=str(args.output), root=str(root))
    except typst.TypstError as error:
        sys.exit(f'{source}: {failing_formula(found, args.work) or error}')
    print(f'{args.output}: {len(chapters)} chapters, {len(found)} formulas, '
          f'{args.output.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
