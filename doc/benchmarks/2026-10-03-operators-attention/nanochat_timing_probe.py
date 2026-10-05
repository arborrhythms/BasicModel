"""Profile the unmodified native evaluator; two controls, no optimizer work."""
import cProfile
import pstats
import time
import eval_nanochat_grammar as gate


def test_profile_native_controls():
    model, data = gate.build_eval_model(gate.DEFAULT_MODEL, autoload=False)
    item = gate.load_manifest(gate.DEFAULT_MANIFEST)['items'][0]
    profile = cProfile.Profile()
    with gate.frozen_online_learning(model):
        profile.enable()
        for control in ('intact', 'shuffled'):
            start = time.perf_counter()
            gate._score_control_batch(model, data, [item], control, 16)
            print(control, time.perf_counter()-start, flush=True)
        profile.disable()
    pstats.Stats(profile).sort_stats('cumtime').print_stats(35)
