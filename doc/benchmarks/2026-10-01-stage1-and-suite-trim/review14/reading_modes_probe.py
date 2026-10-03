"""Retained configurations must select the one current reading mode."""
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[4]
retired={'radix','bpe','mphf','lexicon','none','byte','raw','word','sentence','grammatical'}
old=[]
for path in sorted((ROOT/'data').rglob('*.xml')):
    tree=ET.parse(path)
    for tag in ('./PartSpace/synthesis','./WholeSpace/analysis'):
        value=tree.findtext(tag)
        if value and value.strip() in retired:old.append((str(path.relative_to(ROOT)),tag,value))
assert not old, old
assert ET.parse(ROOT/'data/model.xml').findtext('./WholeSpace/analysis')=='meronomy'
