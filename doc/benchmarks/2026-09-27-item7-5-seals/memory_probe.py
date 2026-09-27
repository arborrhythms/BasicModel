"""Read-only allocation diagnosis of the unchanged two-epoch word-store gate."""
from functools import wraps
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
import resource
import Models
import SentenceCompose
import Language
from test_word_store import test_two_epoch_training_severs_cross_batch_graph
from util import init_device
init_device('cpu')


def report(label, *details):
    print(label, 'peak_GiB', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 3),
          *details, flush=True)


def instrument(name, owner=Models.BasicModel):
    original = getattr(owner, name)
    @wraps(original)
    def call(self, *args, **kwargs):
        report('ENTER ' + name, getattr(args[0], 'shape', '') if args else '')
        result = original(self, *args, **kwargs)
        report('EXIT ' + name)
        return result
    setattr(owner, name, call)


for name in ('getOptimizer', '_run_sentence_batch', '_run_batch_once',
             '_per_word_prelude', '_run_tensor_peer_word_pipeline',
             '_run_sealed_word_bricks', '_per_word_body_step',
             '_sentence_train_step', '_sentence_path_cost', '_publish_sentence_scratch'):
    instrument(name)
instrument('choose_operation', Language.LanguageSpace)
instrument('_stacked_reduced', Language.OperationSelectionLayer)
instrument('_stacked_applied', Language.OperationSelectionLayer)

original = SentenceCompose._SavedValue.__init__
def saved(self, value, storage, copy):
    if value.numel() * value.element_size() > 32 * 2**20:
        report('SAVE', tuple(value.shape), str(value.dtype))
    original(self, value, storage, copy)
SentenceCompose._SavedValue.__init__ = saved

report('START')
test_two_epoch_training_severs_cross_batch_graph()
report('COMPLETE')
