"""One-use source surgery for §11.4; preserve text outside retired branches."""
import ast
from pathlib import Path

class PropertyDecision(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if node.attr in ('property_basis', 'wholePropertyBasis'):
            return ast.Constant(True)
        return self.generic_visit(node)
    def visit_Name(self, node):
        return ast.Constant(True) if node.id == 'property_basis' else node
    def visit_Call(self, node):
        if (isinstance(node.func, ast.Name) and node.func.id == 'getattr' and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in ('property_basis', 'wholePropertyBasis')):
            value = node.args[0]
            if isinstance(value, ast.Name) and value.id in ('sp', '_sp'):
                # Generic space iteration still distinguishes WholeSpace from CS/PS.
                return ast.Call(ast.Name('isinstance', ast.Load()), [value, ast.Name('WholeSpace', ast.Load())], [])
            return ast.Constant(True)
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == 'bool' and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)):
            return ast.Constant(bool(node.args[0].value))
        return node
    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.Constant(not node.operand.value)
        return node
    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        and_op = isinstance(node.op, ast.And)
        kept = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                if bool(value.value) != and_op:
                    return ast.Constant(not and_op)
            else:
                kept.append(value)
        return ast.Constant(and_op) if not kept else kept[0] if len(kept) == 1 else ast.BoolOp(node.op, kept)
    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        return node


def rewrite(path):
    source = path.read_text()
    changed = 0
    while True:
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        starts = [0]
        for line in lines:
            starts.append(starts[-1] + len(line))
        def offset(node, end=False):
            return starts[node.end_lineno - 1 if end else node.lineno - 1] + (node.end_col_offset if end else node.col_offset)
        candidates = []
        for node in ast.walk(tree):
            target = node.test if isinstance(node, ast.If) else node if isinstance(node, ast.IfExp) else None
            if target is None:
                continue
            old = source[offset(target):offset(target, True)]
            if not ('property_basis' in old or 'wholePropertyBasis' in old):
                continue
            original = ast.parse('(' + old + ')', mode='eval').body
            before = ast.dump(original)
            new = PropertyDecision().visit(original)
            if ast.dump(new) == before:
                continue
            if isinstance(node, ast.If) and isinstance(new, ast.Constant):
                body = node.body if new.value else node.orelse
                if body:
                    is_elif = lines[node.lineno - 1].lstrip().startswith('elif ')
                    indent = body[0].col_offset - node.col_offset - (4 if is_elif else 0)
                    text = ''.join(line[indent:] if line.strip() else line
                                   for line in lines[body[0].lineno - 1:body[-1].end_lineno])
                    if is_elif:
                        text = ' ' * node.col_offset + 'else:\n' + text
                else:
                    text = ' ' * node.col_offset + 'pass\n'
                candidates.append((starts[node.lineno-1], starts[node.end_lineno], text, node.end_lineno - node.lineno))
            else:
                candidates.append((offset(target), offset(target, True), ast.unparse(new), target.end_lineno-target.lineno))
        if not candidates:
            break
        # Innermost expressions first. Reparse before touching an enclosing branch.
        lo, hi, replacement, _ = min(candidates, key=lambda item: (item[3], item[1]-item[0]))
        source = source[:lo] + replacement + source[hi:]
        changed += 1
    # Removed branches can expose an unconditional return; delete its unreachable tail.
    while True:
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        edits = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for i, child in enumerate(node.body[:-1]):
                if isinstance(child, (ast.Return, ast.Raise)):
                    edits.append((child.end_lineno, node.body[-1].end_lineno))
                    break
        if not edits:
            break
        lo, hi = min(edits, key=lambda item: item[1]-item[0])
        del lines[lo:hi]
        source = ''.join(lines)
    path.write_text(source)
    print(path, changed)

for name in ('Spaces.py', 'Models.py', 'Language.py'):
    rewrite(Path('bin') / name)
