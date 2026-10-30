"""Measure fixed native inventories and their indexed identity audit.

This admission audit never supplies object meanings or trains a grammar.
Construction/admission failures are reported and re-raised, without resizing.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from corpus_audit import peak_bytes


def allocation_plan(config, output):
    """A lower bound from the constructor's declared shapes, without allocating.

    Empty cache reservations need not be physically resident. This report
    deliberately distinguishes their nominal storage from measured footprint.
    """
    import torch
    from util import init_config, TheXMLConfig, ProjectPaths
    from WordIdentity import resolve_identity_layout
    from architecture import canonical_shape
    from Spaces import _bridge_index_layout
    init_config(path=str(config), defaults_path=str(Path(ProjectPaths.DATA_DIR) / 'model.xml'))
    cfg = TheXMLConfig.data
    resolve_identity_layout(cfg)
    arch, ps, cs, ws = (cfg[key] for key in ('architecture', 'PartSpace', 'ConceptualSpace', 'WholeSpace'))
    if arch['conceptBinding'] != 'aligned' or not arch['serialObjectMeta']:
        raise ValueError('this allocation plan requires the aligned serial corpus architecture')
    capacity, stages = cfg['SymbolSpace']['ltmCapacity'], max(1, arch['bindingDepth'])
    count, parts, wholes = cs['nVectors'], ps['nVectors'], ws['nVectors']
    form = ps['identityPairDim'] + ps['identityLengthDim'] + ps['propertyDim']
    blocks = count * ((parts + 31)//32 + (wholes + 31)//32)
    bridge_capacity = int(cs.get('primingBridgeCapacity', 0) or 0)
    if bridge_capacity:
        blocks = min(blocks, bridge_capacity)
    source_type, _ = _bridge_index_layout(count)
    target_type, _ = _bridge_index_layout((max(parts, wholes) + 31)//32)
    item = lambda dtype: torch.empty((), dtype=dtype).element_size()
    rows = []
    def add(name, shape, size, copies, source):
        rows.append(dict(name=name, shape=shape, dtype_bytes=size, copies=copies,
                         bytes=math.prod(shape)*size*copies, source=source))
    add('priming bridge blocks', [blocks], item(source_type)+item(target_type)+4,
        stages, 'Spaces.py ConceptualSpace.__init__: three counted-prefix buffers')
    add('first equality witness snapshots',
        [cfg['SymbolSpace'].get('equalityWitnessCapacity',8192), cs['nDim']-sum(canonical_shape('ConceptualSpace'))], 4, stages,
        'Layers.py ConceptAllocator.__init__: singleton_values')
    add('shared relation records', [cfg['SymbolSpace'].get('relationCapacity',8192)],
        16+2+32+8+8+4, stages, 'LTM1: native handles, kinds, bands, evidence, accesses and surprise')
    add('shared conceptual dictionary', [count, cs['nDim']], 4, 1,
        'Spaces.py ConceptualSpace.__init__: similarity_codebook')
    add('shared derived placements', [count, cs['meaningWidth']//2], 4, 1,
        'MereologicalCodes.py: placement')
    add('shared cached forms', [count, form], 4, 1, 'MereologicalCodes.py: _forms')
    add('concept priming ledgers', [arch['training']['batchSize'], count], 4, 2*stages,
        'Spaces.py Space.__init__: seen and propagated')
    add('percept prototypes and fixed copies', [parts, form], 4, 2,
        'RadixLayer codebook W and fixed_row_codes')
    add('retained LTM0 ideas', [capacity, cs['nDim']], 4, 1,
        'Layers.py LTM0.__init__: slots; excludes other columns')
    add('recurrence reading summaries', [int(cs.get('recurrenceCapacity', count)), cs['nDim']], 4, 1,
        'Layers.py LTM0.__init__: recurrence_codes')
    report = dict(kind='source-derived nominal storage lower bound, not resident memory',
        config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(), rows=rows,
        total_nominal_bytes=sum(row['bytes'] for row in rows),
        excluded='other buffers, parameters, metadata, optimizer, temporaries, activations, corpus lists')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


def tensor_inventory(module):
    seen, rows = set(), []
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if tensor.layout.__str__() != 'torch.strided':
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key in seen or not storage.nbytes():
            continue
        seen.add(key)
        rows.append(dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                         bytes=storage.nbytes(), parameter=tensor.requires_grad))
    return rows


def measure(config, forms, output):
    import torch
    from util import TheXMLConfig, init_config, ProjectPaths
    from WordIdentity import resolve_identity_layout, WordIdentity
    from Layers import RadixLayer
    from Spaces import WholeSpace, _ladder_periods
    from architecture import canonical_shape
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    init_config(path=str(config), defaults_path=str(Path(ProjectPaths.DATA_DIR) / 'model.xml'))
    cfg = TheXMLConfig.data
    resolve_identity_layout(cfg)
    ps, ws = cfg['PartSpace'], cfg['WholeSpace']
    width = ps['identityPairDim'] + ps['identityLengthDim'] + ps['propertyDim']
    store = RadixLayer(width, initial_cap=ps['nVectors'], word_bounded=True)
    bank = WordIdentity(store, pair_dim=ps['identityPairDim'], ones=ps['identityOnes'],
                        length_dim=ps['identityLengthDim'], property_width=ps['propertyDim'])
    store.identity = bank
    whole = WholeSpace([ws['nInput'], ws['nInputDim']],
                       [ws['nVectors'], ws['nDim'] - sum(canonical_shape('WholeSpace'))],
                       [ws['nOutput'], ws['nOutputDim']])
    bank.refresh_property_fills(whole.subspace.what.primitive_properties)
    pointers = store._basis.W.data_ptr(), store._basis.fixed_row_codes.data_ptr()
    report = dict(config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        forms_sha256=hashlib.sha256(Path(forms).read_bytes()).hexdigest(),
        resolved=cfg, ps_capacity=store.capacity, ps_width=width,
        ladder_periods=_ladder_periods(cfg['SymbolSpace']['ltmCapacity']),
        tensor_inventory=tensor_inventory(store) + tensor_inventory(whole))
    (output / 'construction.json').write_text(json.dumps(report, indent=2) + '\n')
    attempted = 0
    try:
        with Path(forms).open() as source:
            for line in source:
                word = json.loads(line)['form'].encode('ascii')
                attempted += 1
                bank.admit(word)
                if attempted % 10000 == 0:
                    print(json.dumps(dict(words=attempted, rows=len(store), mints=len(bank.mints),
                        seconds=time.monotonic()-started, peak_bytes=peak_bytes())), flush=True)
        audit = bank.audit()
        if pointers != (store._basis.W.data_ptr(), store._basis.fixed_row_codes.data_ptr()):
            raise RuntimeError('native word admission replaced fixed storage')
        report.update(status='complete', words=attempted, rows=len(store),
                      audit=audit, elapsed_seconds=time.monotonic()-started,
                      peak_bytes=peak_bytes(), storage_unchanged=True)
        if audit['collisions'] or audit['reconstruction_errors'] or audit['containment_violations']:
            raise RuntimeError('native indexed identity audit failed')
    except Exception as error:
        report.update(status='failed', attempted=attempted, rows=len(store),
                      words_admitted=len(bank.word_rows), error=str(error),
                      elapsed_seconds=time.monotonic()-started, peak_bytes=peak_bytes())
        (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        raise
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status','words','rows','elapsed_seconds','peak_bytes')}))


def construct(config, output):
    """Load the selected run corpus and construct the complete native model."""
    from util import TheXMLConfig, init_config, ProjectPaths
    from data import TheData
    from Models import BaseModel
    from Spaces import _concept_alloc_of
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    init_config(path=str(config), defaults_path=str(Path(ProjectPaths.DATA_DIR) / 'model.xml'))
    selection = dict(TheXMLConfig.data['architecture']['data'])
    started = time.monotonic()
    TheData.load(selection['dataset'], dat=selection, max_docs=selection['maxDocs'],
                 max_sentence_words=selection['maxSentenceWords'])
    report = dict(config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        corpus=TheData.source_manifest, load_seconds=time.monotonic()-started,
        peak_bytes_after_load=peak_bytes())
    (output / 'loaded.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Constructing the full configured model after loading its selected corpus.', flush=True)
    model = BaseModel.from_config(str(config), data=TheData)[0]
    optimizer = model.getOptimizer(lr=TheXMLConfig.training('learningRate'))
    inventory = tensor_inventory(model)
    # Equality snapshots belong to the plain allocator, outside named_buffers.
    allocators = {id(_concept_alloc_of(cs)): _concept_alloc_of(cs) for cs in model.conceptualSpaces}
    for index, alloc in enumerate(allocators.values()):
        value = alloc.singleton_values
        inventory.append(dict(name=f'allocator.{index}.singleton_values', shape=list(value.shape),
                              dtype=str(value.dtype), bytes=value.untyped_storage().nbytes(), parameter=False))
    enc = model._concept_owner().subspace.whenEncoding
    report.update(status='constructed', tensors=inventory,
        tensor_storage_bytes=sum(row['bytes'] for row in inventory),
        optimizer_initial_state_entries=len(optimizer.state),
        optimizer_note='Adam moments are lazy; constructor success does not certify a training brick',
        clock=dict(long=enc.maxVal, short=enc.period_hf),
        elapsed_seconds=time.monotonic()-started, peak_bytes=peak_bytes())
    (output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('status','tensor_storage_bytes','elapsed_seconds','peak_bytes')}))
    model.End()


def materialize_optimizer(optimizer):
    """Reserve dense moments without changing the native optimizer's update.

    Compact row-local moments are measured after the brick at their native
    touched prefix. Reserving their whole inventory would change the model's
    sparse memory contract. No synthetic gradient or optimizer step is used.
    """
    import torch
    children=getattr(optimizer,'optimizers',None)
    if children is not None:
        return sum(materialize_optimizer(child) for child in children)
    optimizer=getattr(optimizer,'inner',optimizer)
    row_local=bool(getattr(optimizer,'row_local_state',False))
    adam=isinstance(optimizer,(torch.optim.Adam,torch.optim.AdamW,torch.optim.SparseAdam))
    sgd=isinstance(optimizer,torch.optim.SGD)
    if not row_local and not (adam or sgd):
        raise TypeError(f'unsupported capacity optimizer: {type(optimizer).__name__}')
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if row_local or not parameter.requires_grad or optimizer.state.get(parameter):
                continue
            if sgd and not group['momentum']:
                continue
            state = optimizer.state[parameter]
            if sgd:
                if group['dampening']:
                    raise ValueError('early SGD reservation requires zero dampening to preserve its first update')
                state['momentum_buffer']=torch.zeros_like(parameter,memory_format=torch.preserve_format)
            else:
                state['step'] = (0 if isinstance(optimizer,torch.optim.SparseAdam) else
                    torch.zeros((), dtype=torch.float32,
                        device=parameter.device if group.get('capturable') or group.get('fused') else 'cpu'))
                state['exp_avg'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                if group.get('amsgrad'):
                    state['max_exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
    return sum(value.numel()*value.element_size() for state in optimizer.state.values()
               for value in state.values() if torch.is_tensor(value))


def training_inventory(model):
    from Spaces import _concept_alloc_of
    rows = tensor_inventory(model)
    seen = set()
    for index, space in enumerate(model.conceptualSpaces):
        alloc = _concept_alloc_of(space)
        if id(alloc) in seen:
            continue
        seen.add(id(alloc))
        value = alloc.singleton_values
        rows.append(dict(name=f'allocator.{index}.singleton_values', shape=list(value.shape),
                         dtype=str(value.dtype), bytes=value.untyped_storage().nbytes(), parameter=False))
    return rows


def touch_reservations(model):
    """Write every reserved page, preserving its contents and fixed address."""
    import torch
    from Spaces import _concept_alloc_of
    targets = [(name,tensor) for name,tensor in model.named_buffers()
               if any(key in name for key in ('_priming_bridge_', 'recurrence_'))]
    seen = set()
    for index, space in enumerate(model.conceptualSpaces):
        alloc = _concept_alloc_of(space)
        if id(alloc) not in seen:
            seen.add(id(alloc)); targets.append((f'allocator.{index}.singleton_values',alloc.singleton_values))
    report=[]
    with torch.no_grad():
        for name,value in targets:
            pointer=value.data_ptr()
            # Bytewise identity writes preserve even uninitialized integer
            # reservations and floating NaN payloads without dtype arithmetic.
            value.reshape(-1).view(torch.uint8).bitwise_or_(0)
            if value.data_ptr()!=pointer:
                raise RuntimeError('touch changed reservation storage')
            report.append(dict(name=name,bytes=value.numel()*value.element_size(),shape=list(value.shape)))
    return report


def populated_cursor(dataset):
    """First native stage whose document partition populates every stream.

    This is a construction diagnostic entry point, not a claim to have
    trained the skipped prefix. The actual corpus run starts at row zero.
    """
    for index,(lo,hi,windows) in enumerate(dataset._stage_windows):
        if all(end>start for start,end in windows):
            state=dict(dataset.cursor_state(),stage=index,steps=None)
            return state,dict(stage=index,start=lo,end=hi,windows=windows,
                             skipped_training_sentences=lo)
    raise ValueError('declared corpus has no native stage with every stream populated')


def train_brick(config, output, *, complete_prefix=False):
    """A real fixed B28/W256 forward, owned backward, and optimizer step.

    The first fully populated native stage is the search workload; its full
    training vocabulary and that of earlier stages is admitted without
    training a skipped sentence. The selected run starts at the beginning
    and admits each later stage at its boundary.
    No test/validation spelling enters the training vocabulary.
    """
    import torch
    from util import TheXMLConfig, init_config, ProjectPaths, TheCompileBackend
    from data import TheData
    from Models import BaseModel
    from Corpus import admit_training_forms, file_digest
    from Spaces import _concept_alloc_of
    from RowTime import HORIZON, encoder
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    started=time.monotonic(); model=None
    report=dict(status='started',config=str(Path(config).resolve()),
        config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        complete_prefix=complete_prefix,compile_backend=TheCompileBackend,device='cpu')
    save('result.json',report)
    try:
        init_config(path=str(config),defaults_path=str(Path(ProjectPaths.DATA_DIR)/'model.xml'))
        cfg=TheXMLConfig.data
        if cfg['architecture']['training']['batchSize']!=28 or cfg['architecture']['serialWordCapacity']!=256:
            raise ValueError('capacity training requires declared B28/W256')
        selection=dict(cfg['architecture']['data'])
        TheData.load(selection['dataset'],dat=selection,max_docs=selection['maxDocs'],
                     max_sentence_words=selection['maxSentenceWords'])
        report.update(corpus=TheData.source_manifest,load_seconds=time.monotonic()-started,
                      split_sentences={x:len(getattr(TheData,x+'_input')) for x in ('train','validation','test')})
        save('loaded.json',report)
        model=BaseModel.from_config(str(config),data=TheData)[0]
        optimizer=model.getOptimizer(lr=TheXMLConfig.training('learningRate'));model._optimizer=optimizer
        dataset=TheData.data_loader('train',28).dataset
        if complete_prefix:
            stage=0;lo,hi,_=dataset._stage_windows[stage]
            report['workload']=dict(stage=stage,start=lo,end=hi,skipped_training_sentences=0)
        else:
            cursor,workload=populated_cursor(dataset)
            stage=workload['stage'];lo=0;hi=workload['end']
            model.begin_presentation()
            model._resume_corpus_cursor=cursor
            report['workload']=dict(workload,cursor=cursor,
                description='first native stage with 28 nonempty document windows; admission only before this stage')
        answers=(TheData.text_answers or {}).get('train',())
        admitted=admit_training_forms(model,(dataset.inputs[i] for i in range(lo,hi)),
                                    (answers[i] for i in range(lo,hi) if i<len(answers)))
        model._corpus_vocab_stages={(dataset.source_token,i) for i in range(stage+1)}
        report['admission']=dict(through_stage=stage,sentences=hi-lo,forms=len(admitted),
            forms_sha256=hashlib.sha256(b'\n'.join(raw for raw,_ in admitted)).hexdigest(),split='train')
        report['touched']=touch_reservations(model)
        report['optimizer_bytes_before']=materialize_optimizer(optimizer)
        report['optimizer_reservation']='dense Adam/SGD state reserved at step zero; row-local moments use their native touched prefix and are measured after the brick'
        report['expectation_optimizer_bytes_before']=materialize_optimizer(model.percept_expectation.ensure_optimizer())
        report['tensors']=training_inventory(model)
        report['resolved']=cfg
        save('construction.json',report)
        observations=[]; native=model.runBatch
        def recorded(*args,**kwargs):
            began=time.monotonic()
            source=kwargs.get('source_rows')
            save('attempted-brick.json',dict(brick=len(observations),source_rows=source,
                cursor=getattr(model,'_checkpoint_corpus_cursor',None),
                elapsed_seconds=began-started))
            result=native(*args,**kwargs)
            valid=model.inputSpace._word_active_mask
            occupancy=valid.sum(-1).detach().cpu().tolist()
            if valid.shape!=(28,256) or (not complete_prefix and not all(n>0 for n in occupancy)):
                raise RuntimeError(f'capacity brick is not populated B28/W256: {tuple(valid.shape)}, {occupancy}')
            record=dict(brick=len(observations),elapsed_seconds=time.monotonic()-began,
                occupancy=occupancy,sentences=0 if source is None else sum(map(len,source)),
                peak_bytes=peak_bytes(),source_rows=source,
                priming_bridge_used=[list(cs._priming_bridge_counts) for cs in model.conceptualSpaces],
                percept_updates=int(model.percept_expectation.updates))
            observations.append(record)
            with (output/'bricks.jsonl').open('a') as target:target.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            return result
        model.runBatch=recorded
        model.runEpoch(optimizer=optimizer,batchSize=28,max_batches=None if complete_prefix else 1)
        if not observations:raise RuntimeError('capacity measurement processed no brick')
        owner=model._concept_owner(); alloc=_concept_alloc_of(owner)
        identity=model.perceptualSpace.percept_store.identity
        audit=identity.audit()
        if any(audit[key] for key in ('collisions','reconstruction_errors','containment_violations')):
            raise RuntimeError('native admitted-inventory R=0 audit failed')
        clock=encoder()
        checkpoint_record=None
        if complete_prefix:
            checkpoint=output/'continuation.ckpt';model.save_weights(str(checkpoint))
            checkpoint_record=dict(path=str(checkpoint),sha256=file_digest(checkpoint),
                cursor=getattr(model,'_checkpoint_corpus_cursor',None))
        report.update(status='complete',bricks=len(observations),observations=observations,
            optimizer_bytes_after=materialize_optimizer(optimizer),
            expectation_optimizer_bytes_after=materialize_optimizer(model.percept_expectation.ensure_optimizer()),
            tensors=training_inventory(model),rung0_audit=audit,
            occupied=dict(percept=len(model.perceptualSpace.percept_store),words=len(identity.word_rows),
                conceptual_next=alloc.next_id,relation_highest=int(alloc.parts.count)-1,
                equality_witnesses=len(alloc.singleton_members)),
            clock=dict(row_time_horizon=HORIZON,row_periods=[clock.maxVal,clock.period_hf],
                processing_time=float(model.when_time),
                when_periods=[owner.subspace.whenEncoding.maxVal,owner.subspace.whenEncoding.period_hf]),
            checkpoint=checkpoint_record,
            elapsed_seconds=time.monotonic()-started,peak_bytes=peak_bytes())
        save('result.json',report)
        return report
    except Exception as error:
        report.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-started,peak_bytes=peak_bytes())
        save('result.json',report)
        raise
    finally:
        if model is not None:model.End()


def process_footprint(pid):
    """macOS charged footprint, including compressed memory omitted by RSS.

    Layout: SDK sys/resource.h rusage_info_v4, queried through libproc.
    The lifetime maximum catches an allocation spike between guard samples.
    """
    import ctypes,errno
    fields=('user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size '
        'resident_size phys_footprint proc_start_abstime proc_exit_abstime '
        'child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups '
        'child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten '
        'cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background '
        'cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated '
        'cpu_time_qos_user_interactive billed_system_time serviced_system_time '
        'logical_writes lifetime_max_phys_footprint instructions cycles billed_energy '
        'serviced_energy interval_max_phys_footprint runnable_time').split()
    class Usage(ctypes.Structure):
        _fields_=[('uuid',ctypes.c_uint8*16)]+[(name,ctypes.c_uint64) for name in fields]
    library=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
    call=library.proc_pid_rusage;call.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];call.restype=ctypes.c_int
    value=Usage()
    if call(pid,4,ctypes.byref(value)):
        code=ctypes.get_errno()
        if code==errno.ESRCH:return None
        raise OSError(code,'cannot measure process footprint',pid)
    return dict(footprint=value.phys_footprint,lifetime_peak=value.lifetime_max_phys_footprint)


def guarded_trial(command, output, *, memory_bytes=28*1024**3, timeout=3600):
    """Measure aggregate child-tree RSS; kill only this trial on its guard."""
    import os,signal,subprocess,sys
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    environment=dict(os.environ)
    for name in ('RUN_SLOW','BASIC_SEED'):environment.pop(name,None)
    environment.update(BASICMODEL_DEVICE='cpu',OMP_NUM_THREADS='1')
    declaration=dict(command=command,cwd=str(Path.cwd()),memory_bytes=memory_bytes,timeout=timeout,
                     macos_guard='RSS plus charged physical footprint (including compressed memory); lifetime peak catches inter-sample spikes',
                     environment={k:environment.get(k) for k in ('BASICMODEL_DEVICE','OMP_NUM_THREADS','RUN_SLOW','BASIC_SEED')})
    (output/'command.json').write_text(json.dumps(declaration,indent=2)+'\n')
    start=time.monotonic();peak=0;footprint_peak=0;reason=None
    def terminate(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:
            return
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
    with (output/'stdout.log').open('w') as log, (output/'resources.jsonl').open('w') as samples:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
        def sample_trial():
            nonlocal peak,footprint_peak,reason
            listing=subprocess.run(['ps','-axo','pid=,ppid=,pgid=,rss='],capture_output=True,text=True,check=True)
            table={int(row[0]):tuple(map(int,row[1:])) for line in listing.stdout.splitlines()
                   if len(row:=line.split())==4}
            descendants={process.pid}|{pid for pid,(_,group,_) in table.items() if group==process.pid}
            while True:
                expanded=descendants|{pid for pid,(parent,_,_) in table.items() if parent in descendants}
                if expanded==descendants:break
                descendants=expanded
            rss=sum(table[pid][2]*1024 for pid in descendants if pid in table)
            peak=max(peak,rss)
            sample=dict(seconds=time.monotonic()-start,rss=rss,processes=sorted(descendants))
            if sys.platform=='darwin':
                footprints={pid:value for pid in descendants if (value:=process_footprint(pid)) is not None}
                sample['footprints']=footprints
                sample['aggregate_footprint']=sum(value['footprint'] for value in footprints.values())
                footprint_peak=max(footprint_peak,sample['aggregate_footprint'],
                    max((value['lifetime_peak'] for value in footprints.values()),default=0))
                pressure=subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True)
                sample['pressure']=pressure.stdout.strip() if pressure.returncode==0 else None
                swap=subprocess.run(['sysctl','-n','vm.swapusage'],capture_output=True,text=True,check=True)
                sample['swap']=swap.stdout.strip()
            samples.write(json.dumps(sample)+'\n');samples.flush()
            if max(rss,footprint_peak)>memory_bytes:reason='memory_guard'
            elif time.monotonic()-start>timeout:reason='timeout_inconclusive'
            if reason:
                terminate(process)
        try:
            while process.poll() is None:
                sample_trial()
                if reason:break
                time.sleep(1)
        except BaseException as error:
            terminate(process)
            (output/'guard-error.json').write_text(json.dumps(dict(error=repr(error),
                child_pid=process.pid,terminated=True),indent=2)+'\n')
            raise
        code=process.wait()
    result=dict(exit_code=code,status=reason or ('complete' if code==0 else 'failed'),
                peak_process_tree_rss=peak,peak_charged_footprint=footprint_peak,
                elapsed_seconds=time.monotonic()-start)
    (output/'guard.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def search(config, grid, output):
    """Sequential lexicographic bracketing/refinement on the declared grid.

    A non-memory construction/learning error or timeout stops the search;
    it cannot become a claim that a smaller configuration fits.
    """
    import sys
    import xml.etree.ElementTree as ET
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    declaration=json.loads(Path(grid).read_text())
    axes=declaration['axes']; current=dict(declaration['floors']); trials=[]
    if [x['name'] for x in axes]!=['inventory','relationCapacity','equalityWitnessCapacity','primingBridgeCapacity','recurrenceCapacity']:
        raise ValueError('capacity axes must follow the approved lexicographic order')
    (output/'grid.json').write_text(json.dumps(declaration,indent=2)+'\n')
    def trial(axis,value):
        values=dict(current,**{axis:value});tree=ET.parse(config);root=tree.getroot()
        def set_value(section,key,value):
            node=root.find(section)
            if node is None:node=ET.SubElement(root,section)
            child=node.find(key)
            if child is None:child=ET.SubElement(node,key)
            child.text=str(value)
        for section,key,value in [('PartSpace','nVectors',values['inventory']),
            ('ConceptualSpace','nVectors',4*values['inventory']),('SymbolSpace','ltmCapacity',4*values['inventory']),
            ('ConceptualSpace','activeVectors',min(declaration['active_vectors_ceiling'],4*values['inventory'])),
            ('SymbolSpace','relationCapacity',values['relationCapacity']),
            ('SymbolSpace','equalityWitnessCapacity',values['equalityWitnessCapacity']),
            ('ConceptualSpace','primingBridgeCapacity',values['primingBridgeCapacity']),
            ('ConceptualSpace','recurrenceCapacity',values['recurrenceCapacity'])]:set_value(section,key,value)
        name=f'{len(trials):03d}-{axis}-{values[axis]}'
        cfg=output/(name+'.xml');set_value('architecture','weightsPath',str((output/(name+'-periodic.ckpt')).resolve()))
        tree.write(cfg,encoding='unicode')
        command=[sys.executable,str(Path(__file__).resolve()),'--compile-backend',declaration['compile_backend'],'--config',str(cfg.resolve()),
                 '--output',str((output/(name+'-worker')).resolve()),'--train-brick']
        result=guarded_trial(command,output/name,memory_bytes=declaration['memory_bytes'],timeout=declaration['timeout'])
        result.update(values=values,config=str(cfg),axis=axis)
        trials.append(result);(output/'trials.json').write_text(json.dumps(trials,indent=2)+'\n')
        return result['status']
    for axis in axes:
        name=axis['name'];values=axis['values']
        if values!=sorted(set(values)) or current[name]!=values[0]:raise ValueError('grid must increase from its floor')
        good=-1; bad=len(values);index=0
        while index<len(values):
            status=trial(name,values[index])
            if status=='complete':good=index;index=1 if index==0 else min(len(values),index*2)
            elif status=='memory_guard':bad=index;break
            else:
                result=dict(status=status,axis=name,trials=trials,winner=None)
                (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
        while good>=0 and bad-good>1:
            index=(good+bad)//2
            status=trial(name,values[index])
            if status=='complete':good=index
            elif status=='memory_guard':bad=index
            else:
                result=dict(status=status,axis=name,trials=trials,winner=None)
                (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
        if good<0:
            result=dict(status='no_fitting_floor',axis=name,trials=trials,winner=None)
            (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
        current[name]=values[good]
    status=trial('recurrenceCapacity',current['recurrenceCapacity'])
    result=dict(status=status,winner=current if status=='complete' else None,trials=trials,
                winner_config=trials[-1]['config'] if status=='complete' else None)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--compile-backend',default=None)
    parser.add_argument('--forms', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--construct-model', action='store_true')
    parser.add_argument('--train-brick', action='store_true')
    parser.add_argument('--complete-prefix', action='store_true')
    parser.add_argument('--search-grid', type=Path)
    parser.add_argument('--allocation-plan', action='store_true')
    args = parser.parse_args()
    if args.compile_backend is not None:
        import util
        util.init_compile_backend(args.compile_backend)
    if sum((args.construct_model,args.allocation_plan,args.train_brick,args.complete_prefix,args.search_grid is not None))>1:
        parser.error('select one measurement operation')
    if args.search_grid is not None:
        result=search(args.config,args.search_grid,args.output)
        if result['status']!='complete':raise SystemExit(1)
        return
    if args.train_brick or args.complete_prefix:
        train_brick(args.config,args.output,complete_prefix=args.complete_prefix)
        return
    if args.allocation_plan:
        allocation_plan(args.config, args.output)
    elif args.construct_model:
        construct(args.config, args.output)
    elif args.forms is not None:
        measure(args.config, args.forms, args.output)
    else:
        parser.error('--forms or --construct-model is required')


if __name__ == '__main__':
    main()
