"""Part 4b: serial readings always own their tied input reconstruction."""
from pathlib import Path
import json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'suite-trim'))
from port_ledger import definitions
changes=[]
def save(path,old,new):
    assert old!=new,path
    a,b=definitions(old),definitions(new)
    bodies=[dict(name=k,old=a.get(k),new=b.get(k)) for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k) and '::' in k]
    changes.append(dict(file=path,old_source=old,new_source=new,bodies=bodies))
    (ROOT/path).write_text(new)
p=ROOT/'bin/Models.py';old=p.read_text();s=old
start=s.index('        # Stable construction/reconstruction split.')
end=s.index('        self._recon_keep_ideas',start)
s=s[:start]+'''        # Reconstruction belongs to every understanding. Serial grammar
        # readings use the tied sentence inverse; parallel readings already
        # reverse their owned conceptual state at the batch boundary.
        # This is a derived execution property, no longer an XML opt-in.
        self.reconstruct_in_loop = bool(self.serial)
        self.detached_reverse = False
'''+s[end:]
start=s.index('        if self.reconstruct_in_loop and self.detached_reverse:')
end=s.index('        # The global projection',start)
s=s[:start]+s[end:]
s=s.replace('''        self._sentence_reconstruction = self._sentence_ends and (
            self.reconstruct_in_loop or (self._aligned_serial_word_mode()
                and not self.detached_reverse and self.loss.reconstruction_scale > 0))''','''        self._sentence_reconstruction = self._sentence_ends''',1)
start=s.index('            # Rework B (3): on the PER-WORD grammar path')
end=s.index('            elif (mask_pos is not None',start)
s=s[:start]+'''            self._d3_active = False
            self._d3_word_metric = None
            _isp = self.inputSpace
            _per_word = (self.serial and _isp is not None
                         and getattr(_isp, "_per_word_enabled", False))
            if self.serial:
                # Completion belongs to understanding, before reasoning and
                # output can change staging. Both sentence trials have already
                # trained this objective; the batch consumes its owned result.
                owned = self._last_understanding.input_reconstruction
                if owned is None or not torch.is_tensor(owned.byte_cost):
                    raise RuntimeError("tied reconstruction completed without its byte objective")
                lossIn = owned.byte_cost.mean()
                if not train:
                    inputDataPred = owned.event.detach()
'''+s[end:]
s=s.replace('''            _rev_dedupe = (self.reconstruct_in_loop or
                getattr(self, '_sentence_reconstruction', False) or (train and bool(self._d3_active)))''','''            _rev_dedupe = bool(self.serial)''',1)
start=s.index('            # Method-1 -> Method-2 leaf distillation (snap design doc step')
end=s.index('            # Truth-modulated loss:',start)
s=s[:start]+s[end:]
# Remove the model's old detached-student training entry point. The parked
# chooser class still reads old checkpoint tensors in the migration test.
body=definitions(s)['BasicModel::_detached_reverse_construction_loss'];s=s.replace(body,'',1)
s=s.replace('''        # reverse direction (a tied checkpoint under <detachedReverse>)
        # rebuilds the student fresh through the missing-key path below.''','''        # old student parameters are discarded; every serial reading uses
        # the shared tied inverse.''',1)
s=s.replace('''        """Run the tied traversals when ``<reconstructInLoop>`` is on; else
        zero-shaped placeholders (explicit outputs keep a fixed arity)."""''','''        """Run the serial tied inverse, or publish its explicit deferred inputs."""''',1)
s=s.replace('''        if not getattr(self, "reconstruct_in_loop", False):
            return placeholders + state
''','',1)
s=s.replace('''            return self._reconstruct_sentences''','''            return self._reconstruct_sentences''')
save('bin/Models.py',old,s)
p=ROOT/'bin/Language.py';old=p.read_text();s=old
start=s.index('        # Detached reverse construction student.')
end=s.index('        # 7. Sentence expectation defaults',start)
s=s[:start]+'''        # The input inverse is shared with composition. Old detached-student
        # checkpoint keys are migrated by the model; no student is enlisted.
        self.detached_reverse = False
        self.reverse_chooser = None

'''+s[end:]
save('bin/Language.py',old,s)
for name in ('BasicModel_answers_benchmark','BasicModel_expectation_benchmark'):
 p=ROOT/'data'/f'{name}.xml';old=p.read_text();new=old.replace('<detachedReverse>true</detachedReverse>','<detachedReverse>false</detachedReverse>')
 changes.append(dict(file=str(p.relative_to(ROOT)),old_source=old,new_source=new,reason='The detached reconstruction student is retired; tied input reconstruction is mandatory.'))
 p.write_text(new)
p=ROOT/'data/model.xsd';old=p.read_text();new=old.replace('<xs:element name="reconstructInLoop" type="xs:boolean" minOccurs="0"/>','<xs:element name="reconstructInLoop" type="xs:boolean" fixed="true" minOccurs="0"/>').replace('<xs:element name="detachedReverse" type="xs:boolean" minOccurs="0"/>','<xs:element name="detachedReverse" type="xs:boolean" fixed="false" minOccurs="0"/>')
changes.append(dict(file='data/model.xsd',old_source=old,new_source=new));p.write_text(new)
(HERE/'universal-reconstruction-repair.json').write_text(json.dumps(dict(failing_probe='universal-before/worker-000.log',changes=changes),indent=2)+'\n')
