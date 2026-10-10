"""Discrete candidate reads with ordinary, paired-lane support masks."""
import torch
from AttentionObjective import tolerate_heterogeneity


class FieldTraversal:
    def __init__(self, lanes, where, when, valid, allowance, *, tolerance=1.):
        if (lanes.ndim != 4 or lanes.shape[-1] != 2 or lanes.shape[:2] != valid.shape
                or where.shape != (*valid.shape, 2) or when.shape != where.shape
                or allowance.shape != valid.shape[:1]):
            raise ValueError('candidate support, locations and allowances must align')
        self.support = tolerate_heterogeneity(lanes, tolerance)
        self.where, self.when = where, when
        self.valid = valid & self.support.ne(0).any((-1, -2))
        self.remaining = self.valid.clone()
        self.allowance = allowance.clamp_min(0)
        self.iterations = torch.zeros_like(allowance)
        self.stopped = torch.zeros_like(allowance, dtype=torch.bool)
        self.reads = []
        self.backup = None

    @property
    def active(self):
        return self.remaining.any(-1) & ~self.stopped & (self.iterations < self.allowance)

    def select(self, scores, *, alternative=None, greedy=None, scale=None, replay=None, taught=None):
        """One hard candidate per live row; no derivative through the choice.

        The uniform alternative is credited by the existing proposal-corrected
        paired cost. Exhaustion or the independent allowance ends the read.
        """
        active = self.active
        eligible = self.remaining & active[:, None]
        B, W = eligible.shape
        if scores is None:
            action = eligible.long().argmax(-1)
            action = torch.where(active, action, W)
            probabilities = self.support.new_zeros(B, W + 1)
        else:
            logits = scores.masked_fill(~eligible, -torch.inf)
            logits = torch.where(active[:, None], logits, 0.)
            probabilities = logits.softmax(-1)
            action = torch.where(active, logits.argmax(-1), W)
            if replay is not None:
                action = torch.where(replay, greedy, action)
            if alternative is not None:
                options = eligible.scatter(1, greedy.clamp_max(W-1)[:, None], False)
                count = options.sum(-1)
                draw = (torch.rand(B, device=scores.device) * count).long()
                picked = (options & (options.long().cumsum(-1) == draw[:, None] + 1)).long().argmax(-1)
                taken = alternative & active & (count > 0)
                action = torch.where(taken, picked, action)
                probability = probabilities.gather(1, action.clamp_max(W-1)[:, None]).squeeze(1)
                if bool(taken.any()):
                    prior = self.backup
                    self.backup = dict(rows=taken if prior is None else taken | prior['rows'],
                        probability=torch.where(taken, probability, 0. if prior is None else prior['probability']),
                        scale=torch.where(taken, count.to(scores) * scale, 0. if prior is None else prior['scale']),
                        item=torch.where(taken, action, -1 if prior is None else prior['item']))
        lesson_cost = None
        if taught is not None:
            if taught.shape != eligible.shape or bool((taught & ~eligible).any()):
                raise ValueError('the lesson must name an unread supported candidate')
            if not torch.equal(taught.sum(-1), active.long()):
                raise ValueError('the lesson must name exactly one part per active row')
            target = taught.long().argmax(-1)
            if scores is not None:
                lesson_cost = torch.where(active,
                    -logits.log_softmax(-1).gather(1, target[:, None]).squeeze(1), 0.)
                action = torch.where(active, target, W)
        admitted = eligible & (torch.arange(W, device=eligible.device)[None] == action[:, None])
        read = admitted.any(-1)
        self.reads.append(dict(eligible=eligible, admitted=admitted, action=action,
            probabilities=probabilities, active=active,
            alternatives=((eligible.sum(-1) - 1).clamp_min(0) if scores is not None else torch.zeros_like(action))))
        if taught is not None:
            self.reads[-1]['lesson_target'] = taught
            self.reads[-1]['alternatives'] = torch.zeros_like(action)
            if lesson_cost is not None:
                self.reads[-1]['lesson_cross_entropy'] = lesson_cost
        self.iterations += read.long()
        self.remaining &= ~admitted
        self.stopped |= active & ~read
        return admitted

    @property
    def admission(self):
        return (self.valid & ~self.remaining).to(self.support)

    def report(self):
        return dict(iterations=self.iterations.detach(), admission=self.admission.detach(),
            eligible=self.valid.detach(), where=self.where.detach(), when=self.when.detach(),
            support=self.support.detach(),
            reads=[{key: value.detach() for key, value in entry.items()
                    if key != 'probabilities'} for entry in self.reads])
