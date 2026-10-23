import ast,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
mode=sys.argv[1];assert mode in ('original','cube')
run=HERE/('echoic-domain-'+mode);run.mkdir(exist_ok=False)
functions={'bin/Models.py':['BasicModel._output_generate_walk'],
 'bin/Language.py':['LanguageSpace.decoder_eligibility','LanguageSpace._bounded_binary_reconstruction','LanguageSpace.reverse_binary_step'],
 'bin/Layers.py':['Ops._conjunction_kernel','Ops._code_direction'],
 'bin/Generative.py':['inverse_menu']}
comparison=[]
for name,wanted in functions.items():
    def read(text):
        found={}
        def scan(nodes,prefix=''):
            for n in nodes:
                if isinstance(n,ast.ClassDef):scan(n.body,prefix+n.name+'.')
                elif isinstance(n,ast.FunctionDef) and prefix+n.name in wanted:
                    found[prefix+n.name]=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
        scan(ast.parse(text).body);return found
    before=read(subprocess.check_output(['git','show','bb0045d7a:'+name],cwd=ROOT,text=True))
    after=read((ROOT/name).read_text());assert set(before)==set(after)==set(wanted)
    comparison.extend(dict(file=name,function=f,before=before[f],after=after[f],unchanged=before[f]==after[f]) for f in wanted)
(run/'production-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
env=dict(os.environ,BASICMODEL_DEVICE='cpu',RUN_SLOW='1',PYTEST_PLUGINS='echoic_domain_probe',
 ECHOIC_REPLAY=str(HERE/'modules-state'),ECHOIC_DOMAIN=mode,ECHOIC_OUTPUT=str(run),
 DEVELOPER_DIR='/Library/Developer/CommandLineTools',PYTHONPATH=os.pathsep.join(map(str,(HERE,ROOT/'bin',ROOT/'test'))))
command=[str(ROOT/'.venv/bin/python'),'-m','pytest','-q','test/test_echoic_decoder.py::test_same_walk_infers_binary_and_unary_without_a_journal','--tb=short']
record=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),mode=mode,command=command,
 evidence='diagnostic replay only; no gate initialization',started=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (run/'console.log').open('x') as log:result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
record.update(exit=result.returncode,finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
(run/'run.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));raise SystemExit(result.returncode)
