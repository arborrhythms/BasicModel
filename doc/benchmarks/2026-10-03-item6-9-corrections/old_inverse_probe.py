import ast, zipfile
from pathlib import Path
def pytest_configure(config):
    import Models
    archive=Path(__file__).with_name("before.zip")
    with zipfile.ZipFile(archive) as z:
        path=next(n for n in z.namelist() if n.endswith("bin/Models.py"))
        source=z.read(path).decode()
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=="BasicModel")
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_reconstruct_sentences")
    namespace=dict(vars(Models))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(archive),"exec"),namespace)
    Models.BasicModel._reconstruct_sentences=namespace[fn.name]
