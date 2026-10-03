"""Build a reviewable reading-mode migration; --apply installs the preview."""
from pathlib import Path
import ast,json,re,shutil,sys,textwrap,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
preview=HERE/'mode-preview';preview.mkdir(exist_ok=True)
changes={}
def between(text,a,b,replacement):
    assert text.count(a)==1,(a,text.count(a))
    start=text.index(a);end=text.index(b,start)
    return text[:start]+replacement+text[end:]
def save(path,text):
    if path.endswith('.py'):ast.parse(text)
    target=preview/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    changes[path]=text
s=(ROOT/'bin/Legacy.py').read_text()
s=between(s,'# ---------------------------------------------------------------------------\n# Parked PartSpace synthesis front ends','class TrueLayer','')
s=s.replace('The canonical model imports this module only when an older PartSpace synthesis\nmode is explicitly requested. The MATLAB-era', 'The live reading does not import this module. The MATLAB-era')
save('bin/Legacy.py',s)
s=(ROOT/'bin/Spaces.py').read_text()
s=s.replace('TheXMLConfig.space(section, "synthesis") or "lexicon"','TheXMLConfig.space(section, "synthesis") or "meronomy"').replace('_requested_synthesis = "lexicon"','_requested_synthesis = "meronomy"')
s=between(s,'        # Mereology is the sole live synthesis path.','        # Radix promotion defaults;', '''        if _requested_synthesis != "meronomy":
            raise ValueError("PartSpace.synthesis must be meronomy; old reading modes are retired")
        self.synthesis_mode = "meronomy"
        self._meronomy = self._mereological = self._meronomy_words = True
''')
s=between(s,'        if self._legacy_synthesis_mode is not None:\n            from Legacy import validate_part_synthesis','        self._recovered_input = None','')
s=s.replace('bpe=(self.synthesis_mode in ("bpe", "mphf")),','bpe=False,')
s=between(s,'        # Radix/meronomy builds the authoritative surface-form PerceptStore.','        # Native prototype storage', '''        # The native reading owns one surface-form store on its Codebook.
        from Layers import RadixLayer as _PerceptStore
        self.percept_store = _PerceptStore(
            self.nDim, initial_cap=max(int(self.nVectors), 1),
            promotion_threshold=self.chunk_promotion_threshold,
            promotion_min_length=self.chunk_promotion_min_length,
            word_bounded=True, basis=self.subspace.what)
''')
s=s.replace('self._optimize_radix_codebook = self.synthesis_mode == "radix"','self._optimize_radix_codebook = True')
s=between(s,'        # Opt-in/opt-out: the frozen-vocab GPU BPE tokenizer','    def _register_requirements(self):','\n')
start=s.index('    def _build_what_basis(self):',s.index('class PartSpace'))
end=s.index('\n    def ',start+5)
old=s[start:end]
body=old[old.index('            cb = Codebook()'):old.index('\n        basis = Embedding()')]
body=textwrap.dedent(body)
# The Codebook body is retained, including parameter registration and surface semantics.
replacement='''    def _build_what_basis(self):
        """The native surface store shares the Space-owned learnable Codebook."""
        if self.model_type != "embedding":
            return None
'''+textwrap.indent(body,'        ')+'\n'
s=s[:start]+replacement+s[end:]
s=between(s,'        if self._legacy_synthesis_mode is not None:\n            from Legacy import embed_part_stem','        # Canonical synthesis=meronomy:','')
s=between(s,'            if self._legacy_synthesis_mode is not None:\n                from Legacy import embed_part_stem',"        if getattr(vspace, '_demuxed', False)",'''            vspace = self._embed_ladder(vspace)
''')
# The one remaining reading publishes ordinary native surfaces.
s=s.replace('getattr(self, "synthesis_mode", None) == "radix"','getattr(self, "synthesis_mode", None) == "meronomy"')
start=s.index('        _a = None',s.index('class WholeSpace'))
end=s.index('        try:\n            _dww',start)
s=s[:start]+'''        try:
            analysis = TheXMLConfig.space(section, "analysis")
        except KeyError:
            analysis = None
        self.analysis_mode = str(analysis or "meronomy").strip().lower()
        if self.analysis_mode != "meronomy":
            raise ValueError("WholeSpace.analysis must be meronomy; old reading modes are retired")
'''+s[end:]
s=between(s,'        mode = getattr(self, "analysis_mode", "byte")\n        if mode != "meronomy":','        if IS_concepts is None:','')
s=between(s,'        if getattr(self, "analysis_mode", "byte") not in (','        tc = getattr(getattr(self, "subspace", None), "what", None)','')
save('bin/Spaces.py',s)
old={'radix','bpe','mphf','lexicon','none','byte','raw','word','sentence','grammatical'}
retired={'data/MM_bpe.xml','data/MM_20M_legacy.xml'}
configs=[]
for p in sorted((ROOT/'data').rglob('*.xml')):
    rel=str(p.relative_to(ROOT));text=p.read_text();ET.fromstring(text)
    if rel in retired:
        dest=HERE/'retired-configurations'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
        configs.append(dict(config=rel,disposition='retired with old reading mode',archive=str(dest.relative_to(ROOT))))
        continue
    def replace(m):return '<'+m[1]+'>meronomy</'+m[1]+'>' if m[2].strip() in old else m[0]
    new=re.sub(r'<(synthesis|analysis)>([^<]*)</\1>',replace,text)
    if rel=='data/model.xml' and ET.fromstring(new).find('./WholeSpace/analysis') is None:
        new=new.replace('  <WholeSpace>','  <WholeSpace>\n    <analysis>meronomy</analysis>',1)
    if new!=text:
        save(rel,new);configs.append(dict(config=rel,disposition='migrated explicit/default reading selection; all other numerical configuration unchanged'))
# Only the two reading-mode enumerations change; numeric representation knobs stay.
s=(ROOT/'data/model.xsd').read_text()
for tag in ('synthesis','analysis'):
    pattern=r'(<xs:element name="'+tag+r'" minOccurs="0">).*?</xs:element>'
    match=re.search(pattern,s,re.S);assert match
    s=s[:match.start()]+f'''<xs:element name="{tag}" minOccurs="0">
        <xs:simpleType>
          <xs:restriction base="xs:string">
            <xs:enumeration value="meronomy"/>
          </xs:restriction>
        </xs:simpleType>
      </xs:element>'''+s[match.end():]
save('data/model.xsd',s)
(HERE/'mode-preview-manifest.json').write_text(json.dumps(dict(files=list(changes),retired=sorted(retired),configurations=configs),indent=2)+'\n')
if '--apply' in sys.argv:
    for rel,text in changes.items():(ROOT/rel).write_text(text)
    for rel in retired:(ROOT/rel).unlink()
print('preview files',len(changes),'retired',sorted(retired),'applied','--apply' in sys.argv)
