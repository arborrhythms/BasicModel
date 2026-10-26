"""Read-only audit of the full conditional bind menu at initialization."""
from contextlib import contextmanager
from unittest.mock import patch
import torch


def retained_bank_distributions(model):
    """Audit the native chooser with guaranteed retained-context coverage.

    An untrained document reader need not construct an individual at all.
    Supply stored context here without training or forcing a chooser action;
    the ordinary document audit remains separate and unchanged.
    """
    language = model.languageSpace
    layer = language.language_layer.operation_layer
    like = next(layer.parameters())
    width = layer.d_model
    records = []
    for order in (0, 1, 2):
        for count in (1, 4, 7):
            # Eight shared chunks admit at most seven off-stack identities
            # beside this mention. All retained candidates enter the menu.
            points = torch.arange(1, (count+1)*width+1, device=like.device,
                                  dtype=like.dtype).reshape(count+1, width)
            points = torch.nn.functional.normalize(points, dim=-1)
            import STMChunks as chunks
            state = chunks.empty(1,8,width,device=like.device,dtype=like.dtype)
            active = torch.ones(1,device=like.device,dtype=torch.bool)
            ids = model._concept_owner().mtm.transient_ids[:count].to(like.device)
            for index in range(count):
                state,_ = chunks.deposit(state,points[index:index+1],active,
                    order=order,identity=ids[index:index+1])
            state,_ = chunks.deposit(state,points[-1:],active,stack=True,order=order)
            from ReferenceContext import ReferenceBank
            bank = ReferenceBank(torch.full((1,1),-1,device=like.device),like.new_zeros(1,1,width),
                torch.zeros(1,1,dtype=torch.bool,device=like.device),torch.zeros(1,1,dtype=torch.bool,device=like.device),
                like.new_zeros(1,width),torch.zeros(1,dtype=torch.bool,device=like.device))
            scope = torch.zeros(1,8,2,dtype=torch.long,device=like.device)
            scope[:,:,1] = -1
            with binding_distributions() as observed, torch.no_grad():
                with patch.object(language,'_reference_bank',bank,create=True):
                    language.choose_operation(state,active,slots=1,sample=False,policy_grad=False,reference_scope=scope)
            assert observed, 'the controlled field must expose its conditional bind menu'
            for item in observed:
                item.update(context='controlled_retained_chunks',stored_count=count,order=order)
                assert set(item['candidate_ids']) == {0,*ids.tolist()}
                assert len(item['alternatives']) == count+1
            records.extend(observed)
    return records


@contextmanager
def binding_distributions():
    from Language import OperationSelectionLayer
    original = OperationSelectionLayer.forward
    records = []

    def observed(module, x, **kwargs):
        data = kwargs.get('reference_data')
        if not data or not any(data.get('binary_bind',())):
            return original(module, x, **kwargs)
        # Observe the existing chooser before occupation/form priors,
        # reduction pressure and the expectation prime are added. Selection
        # still uses the complete ordinary logits; this is only a measurement.
        scored = {}
        def capture(label, function):
            def call(*args, **options):
                result = function(*args, **options)
                scored[label] = result[1].detach().cpu()
                return result
            return call
        with patch.object(module.chooser, 'score_binary',
                capture('binary', module.chooser.score_binary)), \
                patch.object(module.chooser, 'score_unary',
                capture('unary', module.chooser.score_unary)):
            result = original(module, x, **kwargs)
        complete = result[2]['logits'].detach().cpu()
        logits = torch.cat((scored['binary'].flatten(1), scored['unary'].flatten(1),
                            complete.new_zeros(len(x), 1)), -1)
        logits = logits.masked_fill(~torch.isfinite(complete), -torch.inf)
        selected = result[2]['action'].detach().cpu()
        nr, width = len(data['binary_ops']), x.shape[1]
        references = data['binary_refs'].detach().cpu()
        chunks = data['binary_choices'].detach().cpu()
        for row in range(len(x)):
            for position in range(width-1):
                actions = [position*nr+column for column,bind in enumerate(data['binary_bind'])
                           if bind and torch.isfinite(logits[row,position*nr+column])]
                if not actions:
                    continue
                alternatives = [int(chunks[row,position,action%nr,1])+1 for action in actions]
                candidate_ids = [int(references[row,position,action%nr,1]) for action in actions]
                assert len(set(alternatives)) == len(alternatives), 'retained candidate duplicated in softmax'
                assert 0 in alternatives, 'bind dummy is missing'
                if len(alternatives)==1:
                    continue
                probability = logits[row,actions].softmax(0)
                records.append(dict(row=row,position=position,arity=2,
                    operation=module.op_names.index('bind'),role=1,other=-1,
                    alternatives=alternatives,probabilities=probability.tolist(),
                    candidate_ids=candidate_ids,
                    measured_term='chooser_content',
                    complete_probabilities=complete[row,actions].softmax(0).tolist(),
                    selected=int(selected[row]),actions=actions))
        return result

    with patch.object(OperationSelectionLayer, 'forward', observed):
        yield records
