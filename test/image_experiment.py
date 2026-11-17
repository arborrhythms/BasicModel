"""Certificate 68: one declared image pass beside the zero-width intuition control."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
os.environ['BASICMODEL_DEVICE'] = 'cpu'
import numpy as np
import torch
from bounded_tests import source_snapshot
from image_corpus import load
from recon_bench import _build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--source-hash', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(ROOT)
    digest = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != args.source_hash:
        raise ValueError('image measurement source differs from the declared snapshot')
    (args.out/'source.json').write_text(json.dumps(source, indent=2)+'\n')
    torch.set_num_threads(1)
    streams = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
    torch.save(streams, args.out/'first-random-stream.pt')
    train_x, train_y = load('shapes', 'train')
    heldout = {split: load('shapes', split) for split in ('validation', 'test')}
    protocol = dict(config='data/MM_image.xml', device='cpu', presentations=len(train_x), epochs=1,
        batch_size=8, order='image_corpus.load source order', seed='one captured first draw for both arms',
        teacher='ordinary supervised runBatch output cost', heldout_learning=False, bar=None)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    results = []
    for enabled in (True, False):
        torch.set_rng_state(streams['torch']); np.random.set_state(streams['numpy']); random.setstate(streams['python'])
        arm = args.out/('intuit' if enabled else 'zero'); arm.mkdir()
        tree = ET.parse(ROOT/'data/MM_image.xml')
        if not enabled:
            ET.SubElement(tree.getroot().find('ConceptualSpace'), 'intuitHidden').text = '0'
        config = arm/'model.xml'; tree.write(config, encoding='utf-8', xml_declaration=True)
        model, device, lr, _ = _build_model(str(config))
        optimizer = model.getOptimizer(lr=lr)
        owner = model._concept_owner()
        result = dict(intuit=enabled, parameters=0 if owner.intuit_stack is None else
            sum(p.numel() for p in owner.intuit_stack.parameters()), presentations=0, results={})
        started = time.monotonic()
        try:
            model.begin_presentation()
            for offset in range(0, len(train_x), protocol['batch_size']):
                images = train_x[offset:offset+protocol['batch_size']].to(device)
                labels = train_y[offset:offset+protocol['batch_size']].to(device)
                target = torch.nn.functional.one_hot(labels, 10).to(images).unsqueeze(-1)
                measured, _ = model.runBatch(train=True, optimizer=optimizer, batchNum=offset//protocol['batch_size'],
                    batchSize=len(images), split='train', batch_override=(images, target),
                    source_rows=list(range(offset, offset+len(images))))
                result['presentations'] += len(images)
                model.End(); model.symbolSpace.soft_reset()
                (arm/'progress.json').write_text(json.dumps(dict(result, seconds=time.monotonic()-started), indent=2)+'\n')
                if offset % 80 == 0:
                    print(json.dumps(dict(intuit=enabled, presentations=result['presentations'])), flush=True)
            owner._online_learning_frozen = True
            model.eval()
            for split, (all_images, all_labels) in heldout.items():
                predicted = []
                for offset in range(0, len(all_images), protocol['batch_size']):
                    images = all_images[offset:offset+protocol['batch_size']].to(device)
                    labels = all_labels[offset:offset+protocol['batch_size']].to(device)
                    with torch.no_grad():
                        measured, _ = model.runBatch(train=False, batchNum=offset//protocol['batch_size'],
                            batchSize=len(images), split=split,
                            batch_override=(images, torch.nn.functional.one_hot(labels, 10).to(images).unsqueeze(-1)),
                            source_rows=list(range(offset, offset+len(images))))
                    scores = measured.outputPred.reshape(len(images), -1)
                    assert scores.shape[1] == 10
                    predicted.extend(scores.argmax(-1).cpu().tolist())
                    model.End(); model.symbolSpace.soft_reset()
                correct = sum(p == y for p, y in zip(predicted, all_labels.tolist()))
                result['results'][split] = dict(correct=correct, total=len(all_labels),
                    accuracy=correct/len(all_labels), predictions=predicted)
            result['completed'] = True
        except Exception:
            result.update(completed=False, error=traceback.format_exc())
        finally:
            result['seconds'] = time.monotonic()-started
            (arm/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            results.append(result)
            model.End(); model.symbolSpace.soft_reset()
            del model, owner, optimizer
            gc.collect(); torch._dynamo.reset()
    (args.out/'results.json').write_text(json.dumps(dict(protocol=protocol, arms=results,
        unchanged=source_snapshot(ROOT)==source), indent=2)+'\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
