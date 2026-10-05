"""Migrate shipped declarations; preserve inventory, widths, seeds and bars.

Binding depth and concept layers describe existing parameter geometry. They
are not repeat counts: one attention meter governs perceptual operations.
"""
from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[3]
removed=('serial','modeSchedule','subsymbolicLoop','readingAttention','globalAttention','globalAttentionConsume')
changed=[]
for p in sorted((ROOT/'data').rglob('*.xml')):
 s=p.read_text();old=s
 s=s.replace('subsymbolicOrder>','bindingDepth>')
 s=re.sub(r'<symbolicOrder>(\d+)</symbolicOrder>',lambda m:f'<conceptLayers>{int(m[1])+1}</conceptLayers>',s)
 s=s.replace('selectedThoughtBudget>','attentionBudget>')
 for name in removed:s=re.sub(r'<'+name+r'>[^<]*</'+name+r'>','',s)
 for a,b in (('interLossWeight','sentenceExpectationLossWeight'),('armaScale','sentenceExpectationArmaScale'),('interContrastiveWeight','sentenceExpectationContrastiveWeight')):s=s.replace(a+'>',b+'>')
 if s!=old:p.write_text(s);changed.append(str(p.relative_to(ROOT)))
(Path(__file__).parent/'config-migration.json').write_text(json.dumps(dict(paths=changed,policy='Widths, inventory sizes, seeds, resource guards and gate bars unchanged. Existing binding parameters retain their depth; concept layers retain their declared order capacity.'),indent=2)+'\n')
