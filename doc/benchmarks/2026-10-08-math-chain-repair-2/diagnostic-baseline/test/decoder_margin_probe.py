"""Observe decoder logits at their real backward/step, without changing RNG."""
import torch


class DecoderMarginProbe:
    def __init__(self, emit):
        self.emit = emit
        self.serial = 0
        self.step = -1
        self.active = None
        self.ready = []

    def capture(self, model, logits, parent, legal, *, round, explore, live, metadata):
        if int(round) != 0 or not logits.requires_grad:
            return
        language = model.languageSpace
        binary = len(language._generate_binary_ops)
        stop = binary + len(language._generate_unary_ops)
        row = dict(id=self.serial, **metadata, path='explore' if explore else 'greedy',
                   live=live.detach().cpu().tolist(), legal=legal.detach().cpu().tolist(),
                   binary_indices=list(range(binary)), stop_index=stop,
                   binary_rule_ids=language._generate_binary_rule_ids.detach().cpu().tolist(),
                   binary_rule_names=list(language._generate_binary_names), stop_name='STOP',
                   logits=logits.detach().cpu().tolist(),
                   margin=(logits[:, stop:stop+1]-logits[:, :binary]).detach().cpu().tolist())
        self.serial += 1
        self.emit('decoder_first_logits', **row)
        reference = parent.detach().clone()
        def observe(gradient):
            # Diagnostic autograd.grad calls must not masquerade as training.
            if self.active is not None:
                entry = self.active.setdefault(row['id'], dict(row=row,
                    language=language, parent=reference, gradient=torch.zeros_like(gradient), calls=0))
                entry['gradient'].add_(gradient.detach())
                entry['calls'] += 1
        logits.register_hook(observe)

    def begin_backward(self):
        assert self.active is None and not self.ready
        self.step += 1
        self.active = {}

    def end_backward(self):
        self.ready = list(self.active.values())
        self.active = None

    @staticmethod
    def margins(entry):
        logits = entry['language'].generate_policy_logits(entry['parent'])
        row = entry['row']
        return logits[:, row['stop_index']:row['stop_index']+1]-logits[:, row['binary_indices']]

    @torch.no_grad()
    def before_step(self):
        for entry in self.ready:
            entry['before'] = self.margins(entry).detach().clone()

    @torch.no_grad()
    def after_step(self):
        walks = []
        for entry in self.ready:
            row, gradient = entry['row'], entry['gradient']
            stop, binary = row['stop_index'], row['binary_indices']
            after = self.margins(entry)
            walks.append(dict(id=row['id'], epoch=row.get('epoch'), batch=row.get('batch'),
                trial=row.get('trial'), path=row['path'],
                gradient=gradient.cpu().tolist(), gradient_calls=entry['calls'],
                stop_minus_undo_gradient=(gradient[:, stop:stop+1]-gradient[:, binary]).cpu().tolist(),
                fixed_parent_margin_before=entry['before'].cpu().tolist(),
                fixed_parent_margin_after=after.cpu().tolist(),
                fixed_parent_margin_change=(after-entry['before']).cpu().tolist()))
        self.emit('decoder_margin_step', step=self.step, walks=walks)
        self.ready = []
