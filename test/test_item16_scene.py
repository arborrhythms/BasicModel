"""Item 1.6, round one, step A: the scene reaches the model.

The certificates S1 to S4 of §20.4 of
doc/plans/2026-11-18-item1-6-felt-math-world.md, each printing what it
measured. A model whose configuration declares the shape of a scene's box
(``architecture.sceneShape``) has a range ``scene`` of that shape on a
stretch of the first axis of its own; a row may carry a ``scene`` beside its
``text``; a frozen felt document becomes such rows. Nothing reads the scene.
"""
import glob
import json
import os
import subprocess
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_PROJECT, "data")
_SCENE = os.path.join("scene", "BasicModel_scene.xml")
# The configuration the scene's was copied from.
_SOURCE = "BasicModel.xml"
# Every configuration's ranges on the source before the scene's box, each
# with its addresses, its box and a digest of its wires
# (tmp/item1-6/scene_ranges.py, run on the frozen copy).
_BEFORE = os.path.join(_PROJECT, "test", "fixtures", "item16_scene_ranges_before.json")
# The configurations built afresh here, each in a process of its own and
# one at a time: a text line, a numeric line and a numeric box. The others
# do not fit the bounded runner's memory (BasicModel.xml takes about 6.6 GB
# to build, BasicModel_corpus.xml about 16 GB).
_BUILT = ("MM_text.xml", "MM_xor.xml", "MM_image.xml")

# Builds one configuration and prints its ranges as the fixture has them.
_CHILD = r'''
import hashlib, json, os, sys, warnings
warnings.simplefilter("ignore")
import numpy
import Models, Language
from util import init_config
path = sys.argv[1]
init_config(path=path, defaults_path=sys.argv[2])
Language.TheGrammar._configured = False
model, cfg = Models.BasicModel.from_config(path)
reg = model.where_registry
grid = reg.grid.detach().cpu().float().numpy()
ranges = {}
for name, (lo, hi) in reg.slices.items():
    rows = numpy.ascontiguousarray(grid[lo:hi]) if hi > lo else numpy.zeros((0, grid.shape[1]), numpy.float32)
    ranges[name] = dict(addresses=[int(lo), int(hi)],
                        origin=[int(v) for v in reg.boxes[name][0]],
                        shape=[int(v) for v in reg.boxes[name][1]],
                        wires=hashlib.sha256(rows.tobytes()).hexdigest())
print("@@" + json.dumps(dict(ranges=ranges, capacity=int(reg.capacity),
                             period=int(reg.encoding.maxVal))))
'''


def _before():
    with open(_BEFORE, encoding="utf-8") as fh:
        return json.load(fh)


def _configurations():
    """Every configuration under data/, by its path below data/."""
    import xml.etree.ElementTree as ET
    out = []
    for path in sorted(glob.glob(os.path.join(_DATA, "**", "*.xml"), recursive=True)):
        if os.path.basename(path) == "model.xml":
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        if root.find("architecture") is None and root.find("InputSpace") is None:
            continue
        out.append(os.path.relpath(path, _DATA))
    return out


def _ranges(name):
    """One configuration's ranges, built in a process of its own."""
    env = dict(os.environ, PYTHONPATH=os.path.join(_PROJECT, "bin"), BASICMODEL_DEVICE="cpu")
    proc = subprocess.run([sys.executable, "-c", _CHILD, os.path.join(_DATA, name),
                           os.path.join(_DATA, "model.xml")],
                          capture_output=True, text=True, cwd=_PROJECT, env=env)
    line = [l for l in proc.stdout.splitlines() if l.startswith("@@")]
    if not line:
        return dict(error=(proc.stderr.strip().splitlines() or ["(no output)"])[-1])
    return json.loads(line[-1][2:])


def _flat(path):
    """A configuration's elements, each by its path, with its text."""
    import xml.etree.ElementTree as ET
    out = {}
    def walk(node, at):
        for child in node:
            if not isinstance(child.tag, str):
                continue
            key = f"{at}/{child.tag}"
            out[key] = (child.text or "").strip()
            walk(child, key)
    walk(ET.parse(path).getroot(), "")
    return out


def _declared_shape():
    return _flat(os.path.join(_DATA, _SCENE))["/architecture/sceneShape"]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """The text model of the echo test (``echo_model``), as it is and with
    the scene's shape that the scene's configuration declares."""
    from test_item17_echo import echo_model
    with pytest.MonkeyPatch.context() as monkeypatch:
        scene = echo_model(tmp_path_factory.mktemp("scene"), monkeypatch,
                           architecture_overrides={"sceneShape": _declared_shape()})
        plain = echo_model(tmp_path_factory.mktemp("plain"), monkeypatch)
        yield dict(scene=scene, plain=plain)
        scene.End()
        plain.End()


@pytest.fixture(scope="module")
def scene_model(models):
    return models["scene"]


# -- S1: the box ------------------------------------------------------------------

def test_s1_the_key():
    """The key's default declares no box; the scene's configuration alone
    declares one, and differs from the configuration it was copied from in
    that, in its weights' path and in ``createMetaConcepts`` alone."""
    defaults = _flat(os.path.join(_DATA, "model.xml"))
    assert defaults["/architecture/sceneShape"] == ""
    declaring = [n for n in _configurations()
                 if "/architecture/sceneShape" in _flat(os.path.join(_DATA, n))]
    assert declaring == [_SCENE]
    scene, source = _flat(os.path.join(_DATA, _SCENE)), _flat(os.path.join(_DATA, _SOURCE))
    differ = sorted(k for k in set(scene) | set(source) if scene.get(k) != source.get(k))
    assert differ == ["/architecture/createMetaConcepts", "/architecture/sceneShape", "/architecture/weightsPath"]
    assert scene["/architecture/sceneShape"] == "8 8 8"
    print(f"S1 sceneShape: default empty; declared by {declaring} alone, as "
          f"{scene['/architecture/sceneShape']!r}; it differs from {_SOURCE} in {differ}")


def test_s1_the_configurations_are_those_of_before():
    """Every configuration under data/ but the scene's was there before."""
    before = _before()
    now = _configurations()
    assert sorted(set(now) - {_SCENE}) == sorted(before), sorted(set(now) ^ set(before))
    print(f"S1 {len(before)} configurations before; {len(now)} now, the one added {_SCENE}")


@pytest.mark.parametrize("name", _BUILT)
def test_s1_every_range_is_where_it_was(name):
    """A configuration that does not declare the box has no range ``scene``,
    and each of its ranges has the addresses, the box and the wires it had."""
    then, now = _before()[name], _ranges(name)
    assert "error" not in now, (name, now)
    assert "scene" not in now["ranges"]
    assert now == then, name
    print(f"S1 {name}: {len(now['ranges'])} ranges, capacity {now['capacity']}, "
          f"period {now['period']}: addresses, boxes and wires as before")


def test_s1_the_scene_has_its_box(models):
    """With the shape declared the model has a range ``scene`` of 512
    places, 8 by 8 by 8, after every other range and sharing no place with
    one; every other range has the addresses, the box and the wires it has
    without it."""
    from WhereRegistry import WhereRegistry
    scene, plain = models["scene"].where_registry, models["plain"].where_registry
    assert models["scene"].inputSpace.scene_shape == (8, 8, 8)
    assert models["plain"].inputSpace.scene_shape == () and "scene" not in plain.slices
    lo, hi = scene.slices["scene"]
    origin, shape = scene.boxes["scene"]
    assert hi - lo == 512 and shape == (8, 8, 8)
    assert lo == plain.capacity and scene.capacity == plain.capacity + 512
    assert scene.encoding.maxVal == plain.encoding.maxVal
    assert list(scene.slices) == [*plain.slices, "scene"]
    grid, before = scene.grid.detach().cpu(), plain.grid.detach().cpu()
    for name in plain.slices:
        assert scene.slices[name] == plain.slices[name] and scene.boxes[name] == plain.boxes[name], name
        a, b = plain.slices[name]
        assert torch.equal(grid[a:b], before[a:b]), name
    # The registry refuses two boxes that share a place; this one is made
    # again from its own boxes.
    WhereRegistry(tuple((n, scene.boxes[n][1], scene.boxes[n][0]) for n in scene.slices))
    reach = max(o[0] + sh[0] for n, (o, sh) in scene.boxes.items() if n != "scene" and min(sh) > 0)
    assert origin[0] >= reach and origin[1:] == (0, 0)
    print(f"S1 the echo's text model with sceneShape {_declared_shape()!r}: scene at addresses "
          f"{lo}..{hi}, origin {origin}, shape {shape}; the first axis of the other ranges ends "
          f"at {reach}; capacity {plain.capacity} -> {scene.capacity}, period "
          f"{plain.encoding.maxVal} -> {scene.encoding.maxVal}; {', '.join(plain.slices)}: "
          "addresses, boxes and wires as without it")


# -- S2: the row ------------------------------------------------------------------

def test_s2_the_row(scene_model):
    space = scene_model.inputSpace
    rows = [dict(text="nine.", scene=[[1, 2, 0, 200], [7, 7, 7, 1], [0, 0, 0, 255]]),
            dict(text="", scene=[]),
            dict(text="one.", scene=None),
            dict(text="two."),
            dict(text="", scene=[[3, 4, 0, 51]])]
    space.prepInput(rows)
    kept = space._scenes
    assert kept[0] == ((1, 2, 0, 200 / 255), (7, 7, 7, 1 / 255), (0, 0, 0, 1.0))
    assert kept[1] == () and kept[1] is not None
    assert kept[2] is None and kept[3] is None
    assert kept[4] == ((3, 4, 0, 51 / 255),)
    # A row of text alone leaves no scenes kept.
    space.prepInput(["nine."])
    assert space._scenes is None
    space.prepInput([dict(text="nine.", scalar=0.5)])
    assert space._scenes == [None] and space._scalar_values == [0.5]
    print(f"S2 kept: {kept}")


@pytest.mark.parametrize("rows, message", [
    ([dict(text="nine.", picture=[])], r"text, scalar and scene only, not \['picture'\]"),
    ([dict(text="", scene=[[8, 0, 0, 10]])], r"outside the scene's box"),
    ([dict(text="", scene=[[0, 0, -1, 10]])], r"outside the scene's box"),
    ([dict(text="", scene=[[0, 0, 0, 0]])], r"steps outside 1 to 255"),
    ([dict(text="", scene=[[0, 0, 0, 256]])], r"steps outside 1 to 255"),
    ([dict(text="", scene=[[2, 3, 0, 10], [2, 3, 0, 40]])], r"two pixels at the place \(2, 3, 0\)"),
], ids=["key", "outside-x", "outside-z", "steps-0", "steps-256", "one-place"])
def test_s2_what_is_refused(scene_model, rows, message):
    with pytest.raises(ValueError, match=message) as caught:
        scene_model.inputSpace.prepInput(rows)
    print(f"S2 refused: {caught.value}")


def test_s2_a_scene_on_a_model_with_no_box(models):
    space = models["plain"].inputSpace
    with pytest.raises(ValueError, match=r"no box for one") as caught:
        space.prepInput([dict(text="nine.", scene=[])])
    # No image is no scene, on any model.
    space.prepInput([dict(text="nine.", scene=None)])
    assert space._scenes == [None]
    print(f"S2 refused on the model with no box: {caught.value}")


def test_s2_a_row_with_an_image_and_no_words_is_active(scene_model):
    space = scene_model.inputSpace
    space.prepInput([dict(text="", scene=[[1, 1, 0, 100]]), dict(text="", scene=[]),
                     dict(text="", scene=None), dict(text="", scalar=0.5)])
    space._finalize_sentence_word_layout(torch.zeros(4, 3, dtype=torch.bool))
    slot = space._packed_sentence_slot_mask[:, 0].tolist()
    assert slot == [True, True, False, True]
    print(f"S2 a row's sentence slot (an image, an empty image, nothing, a value): {slot}")


# -- S3: the corpus ---------------------------------------------------------------

def test_s3_the_corpus(scene_model):
    import felt_world_corpus as felt
    with open(os.path.join(felt.FELT, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    space = scene_model.inputSpace
    documents = entries = pixels = images = empty = 0
    for entry in manifest["files"]:
        for split in entry["rows"]:
            for document in felt.load(entry["category"], split):
                rows, question, answer = felt.rows(document)
                assert (question, answer) == (document.question, document.answer)
                assert [r["text"] for r in rows] == list(document.sentences)
                space.prepInput(rows)
                assert space._last_sentences == list(document.sentences)
                assert len(space._scenes) == len(rows) == len(document.scenes)
                for kept, scene in zip(space._scenes, document.scenes):
                    if scene is None:
                        assert kept is None
                        continue
                    assert kept is not None
                    back = tuple((x, y, z, round(v * 255)) for x, y, z, v in kept)
                    assert back == scene
                    images += 1
                    empty += not scene
                    pixels += len(scene)
                documents += 1
                entries += len(rows)
    assert len(manifest["files"]) == 27
    print(f"S3 {documents} documents of 27 files, {entries} rows, {images} images "
          f"({empty} empty), {pixels} pixels: taken, and given back exactly")


# -- S4: far enough apart (a measure) ---------------------------------------------

def test_s4_far_enough_apart(scene_model):
    """The least distance between the wires of a place of ``text`` and a
    place of ``scene``, beside the least between two neighbouring places."""
    registry = scene_model.where_registry
    grid = registry.grid.detach().cpu().double()
    tlo, thi = registry.slices["text"]
    slo, shi = registry.slices["scene"]
    text, scene = grid[tlo:thi], grid[slo:shi]
    across = float(torch.cdist(text, scene).min())
    # Neighbours: one step along one axis, in the line of text and in the box.
    steps = [(text[1:] - text[:-1]).norm(dim=-1)]
    place = registry.place(torch.arange(slo, shi))
    for axis in range(place.shape[-1]):
        step = place.clone()
        step[:, axis] += 1
        address = registry.address(step)
        inside = address >= 0
        steps.append((grid[address[inside]] - grid[torch.arange(slo, shi)[inside]]).norm(dim=-1))
    neighbour = float(torch.cat(steps).min())
    origin = registry.boxes["scene"][0]
    print(f"S4 at capacity {registry.capacity} (period {registry.encoding.maxVal}), box origin "
          f"{origin}: least text-to-scene {across:.6f}, least between neighbours {neighbour:.6f}, "
          f"ratio {across / neighbour:.1f}")
    assert across >= neighbour
