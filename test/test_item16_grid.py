"""Item 1.6, §16: the `.where` as a space of three dimensions.

The certificates G1 to G9 of §16.5 of
doc/plans/2026-11-18-item1-6-felt-math-world.md, each printing what it
measured. A place is (x, y, z); its wires are four for each axis; the grid
is one fixed parameter of one row of twelve wires for each address.
"""
import glob
import json
import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_PROJECT, "data")
_BEFORE = os.path.join(_PROJECT, "test", "fixtures", "item16_widths_before.json")

# The greatest registry a test declares (test_coordinate_registry).
GREATEST = 512_808_192
_WIDTHS = ("nDim", "nInputDim", "nOutputDim")
_COUNTS = ("nVectors", "nInput", "nOutput")
# Sections whose declared width is an event, and so embraces the band. A
# concept is the concatenation of a percept's .what, .where and .when
# (sigma-pi §19.133), so the conceptual width embraces it too; the output
# takes a concept in and gives no band out.
_EMBRACING = ("InputSpace", "PartSpace", "ModalSpace", "ConceptualSpace", "WholeSpace")
_BANDED = ("InputSpace", "PartSpace", "ModalSpace", "WholeSpace")
# The packed byte buffer of the three BasicModel configurations, 512 wide, is
# no event: the form layout replaces the input's width, and the part space's
# input reshape divides it (144 * 8192 = 2304 * 512).
_BUFFER = {("InputSpace", "nDim", 512), ("PartSpace", "nInputDim", 512)}


def _before():
    with open(_BEFORE, encoding="utf-8") as fh:
        return json.load(fh)


def _declared(path):
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    out = {}
    for section in root:
        values = {}
        for key in _WIDTHS + _COUNTS:
            node = section.find(key)
            if node is not None and node.text is not None:
                values[key] = int(node.text.strip())
        if values:
            out[section.tag] = values
    return out


def _model(name):
    import Language
    import Models
    from util import init_config
    path = os.path.join(_DATA, name)
    init_config(path=path, defaults_path=os.path.join(_DATA, "model.xml"))
    Language.TheGrammar._configured = False
    model, _cfg = Models.BasicModel.from_config(path)
    return model


@pytest.fixture(scope="module")
def xor_model():
    return _model("MM_xor.xml")


def _spaces(model):
    spaces = [("InputSpace", model.inputSpace), ("PartSpace", model.perceptualSpace)]
    spaces += [("ConceptualSpace[%d]" % i, s) for i, s in enumerate(model.conceptualSpaces)]
    spaces += [("WholeSpace", model.wholeSpace), ("SymbolSpace", model.symbolSpace),
               ("OutputSpace", model.outputSpace)]
    return spaces


def _one_axis(capacity):
    """The wires an address had before this round: the one-axis ladder at
    the registry's capacity (what ``WhereRegistry.code`` was)."""
    from Spaces import WhereEncoding
    return WhereEncoding(nWhere=4, nWhen=4).set_capacity(capacity)


def test_g1_the_width():
    from architecture import AXIS_WIRES, WHERE_AXES, WHERE_WIRES, canonical_shape
    from BasicPercepts import WHERE_WIRES as percept_wires
    from util import TheXMLConfig
    axes = int(TheXMLConfig.get("architecture.whereAxes"))
    print(f"G1 whereAxes={axes} AXIS_WIRES={AXIS_WIRES} WHERE_WIRES={WHERE_WIRES}")
    assert WHERE_AXES == axes == 3 and WHERE_WIRES == 4 * axes == 12 == percept_wires
    assert canonical_shape("PartSpace") == (12, 4)
    # The key is in data/model.xml alone.
    for path in glob.glob(os.path.join(_DATA, "*.xml")):
        if os.path.basename(path) != "model.xml":
            with open(path, encoding="utf-8") as fh:
                assert "whereAxes" not in fh.read(), path
    before = _before()
    configs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(_DATA, "*.xml")))
    assert sorted(before["declared"]) == configs
    grown = 0
    for name in configs:
        now, then = _declared(os.path.join(_DATA, name)), before["declared"][name]
        assert sorted(now) == sorted(then), name
        for section, values in then.items():
            assert sorted(now[section]) == sorted(values), (name, section)
            for key, old in values.items():
                embraces = key in _WIDTHS and old > 0 and (section, key, old) not in _BUFFER and (
                    section in _EMBRACING or (section == "OutputSpace" and key == "nInputDim"))
                assert now[section][key] == old + (8 if embraces else 0), (name, section, key)
                grown += embraces
            if section in _BANDED and values.get("nDim", 0) > 0 and (section, "nDim", values["nDim"]) not in _BUFFER:
                # The .what of a banded space: its event less the band.
                assert now[section]["nDim"] - sum(canonical_shape(section)) == values["nDim"] - 8
    print(f"G1 {len(configs)} configurations; {grown} declared widths grew by 8, "
          "every .what, vector count and place count as before")


@pytest.mark.parametrize("name", ["MM_xor.xml", "MM_image.xml"])
def test_g1_built_widths_are_as_before(name):
    """The built model's spaces against the frozen copy's, from step 0."""
    built = _before()["built"][name]
    model = _model(name)
    rows = {space: (int(s.subspace.nWhat), int(s.subspace.nWhere), int(s.outputShape[0]))
            for space, s in _spaces(model)}
    for row in built["spaces"]:
        what, band, places = rows[row["space"]]
        if row["band"][0]:
            assert what == row["what"] and band == row["band"][0] + 8, row
        else:
            # A concept embraces the percept's band: it grows by 8.
            assert what == row["what"] + (8 if row["space"].startswith("Conceptual") else 0), row
        assert places == row["places"], row
        print(f"G1 {name} {row['space']}: .what {row['what']} -> {what}, places {places}")
    assert {k: list(v) for k, v in model.where_registry.slices.items()} == built["ranges"]
    model.End()


def test_g2_the_first_axis_is_todays(xor_model):
    registry = xor_model.where_registry
    old = _one_axis(registry.capacity)
    addresses = torch.arange(registry.capacity)
    grid = registry.grid.detach().cpu()
    first = old.encode(addresses).float().cpu()
    zero = registry.encoding.encode(torch.tensor(0)).float().cpu()
    torch.testing.assert_close(grid[:, :4], first, rtol=0, atol=0)
    torch.testing.assert_close(grid[:, 4:8], zero.expand(len(addresses), -1), rtol=0, atol=0)
    torch.testing.assert_close(grid[:, 8:], zero.expand(len(addresses), -1), rtol=0, atol=0)
    assert isinstance(registry.grid, torch.nn.Parameter) and registry.grid.requires_grad
    assert not any(p is registry.grid for p in xor_model.parameters())
    assert not hasattr(registry, "code")
    print(f"G2 {registry.capacity} addresses of MM_xor: first four wires equal the one-axis "
          "ladder exactly; the other eight are the code of zero")


def _round_trip(registry, addresses):
    place = registry.place(addresses)
    assert (registry.address(place) == addresses).all()
    wires = registry.encoding.encode_place(place)
    assert (registry.encoding.decode_place(wires) == place).all()
    assert (registry.decode(wires) == addresses).all()
    return place


def test_g3_there_and_back(xor_model):
    from WhereRegistry import WhereRegistry
    registry = xor_model.where_registry
    _round_trip(registry, torch.arange(registry.capacity))
    assert (registry.decode(registry.grid.detach()) == torch.arange(registry.capacity,
                                                                     device=registry.grid.device)).all()
    image = WhereRegistry((("text", 4096), ("numeric", (28, 28)), ("scalar", 1),
                           ("parts", 8), ("wholes", 64), ("conceptual", 8)))
    _round_trip(image, torch.arange(image.capacity))
    great = WhereRegistry((("text", 512_000_000), ("numeric", (28, 28)), ("parts", 808_192 - 784)))
    assert great.capacity == GREATEST
    ends = [v for lo, hi in great.slices.values() for v in (lo, hi - 1)]
    sample = torch.cat((torch.tensor(ends), torch.randint(0, GREATEST, (100_000,))))
    _round_trip(great, sample)
    print(f"G3 every address of MM_xor ({registry.capacity}) and of a registry with a 28 by 28 "
          f"box ({image.capacity}) round-trips; at {GREATEST} addresses the ends of every "
          "range and 100000 drawn addresses round-trip")


def test_g4_boxes():
    from WhereRegistry import WhereRegistry
    registry = WhereRegistry((("text", 100), ("numeric", (28, 28)), ("scalar", 1)))
    lo, hi = registry.slices["numeric"]
    origin = torch.tensor(registry.boxes["numeric"][0])
    for r, c in ((0, 0), (0, 27), (27, 0), (13, 5), (27, 27)):
        address = lo + 28 * r + c
        place = registry.place(torch.tensor(address))
        assert (place - origin).tolist() == [c, r, 0]
        assert int(registry.address(place)) == address
    assert hi - lo == 784
    with pytest.raises(ValueError):
        WhereRegistry((("a", (4, 4)), ("b", (2, 2), (3, 3, 0))))
    WhereRegistry((("a", (4, 4)), ("b", (2, 2), (4, 0, 0))))
    print(f"G4 pixel (r, c) is at origin + (c, r, 0) = {tuple(origin.tolist())} + (c, r, 0), "
          "at its row-by-row address; two boxes sharing a place are refused")


def test_g4_the_numeric_input_is_a_box():
    """install_where_registry gives an image input its box and XOR's two
    places a line."""
    image = _model("MM_image.xml")
    assert image.where_registry.boxes["numeric"][1] == (28, 28, 1)
    image.End()
    xor = _model("MM_xor.xml")
    assert xor.where_registry.boxes["numeric"][1] == (2, 1, 1)
    xor.End()


def test_g5_the_match_on_two_axes():
    from FieldReverse import where_match
    from WhereRegistry import WhereRegistry
    registry = WhereRegistry((("text", 100), ("numeric", (28, 28)), ("scalar", 1)))
    lo, hi = registry.slices["numeric"]
    grid = registry.grid.detach().cpu()[lo:hi]
    q = where_match(grid, grid, registry.axes, 1000.)
    assert (q.argmax(-1) == torch.arange(hi - lo)).all()
    distance = lambda a, b: sum((grid[a, s:e] - grid[b, s:e]).norm() for s, e in registry.axes)
    steps = []
    for r in range(1, 27):
        for c in range(1, 27):
            at = 28 * r + c
            sides = [distance(at, at + 1), distance(at, at - 1), distance(at, at + 28), distance(at, at - 28)]
            steps.append(torch.stack(sides))
            torch.testing.assert_close(torch.stack(sides), sides[0].expand(4), rtol=0, atol=1e-6)
    steps = torch.stack(steps)
    print(f"G5 every place of a 28 by 28 box matches itself best; a step in x and a step in y "
          f"are equally far: {float(steps.min()):.6f} to {float(steps.max()):.6f}")


def test_g6_text_lies_on_the_first_axis(xor_model):
    registry = xor_model.where_registry
    lo, hi = registry.slices["text"]
    place = registry.place(torch.arange(lo, hi))
    assert (place[:, 0] == torch.arange(lo, hi, device=place.device)).all()
    assert (place[:, 1:] == 0).all()
    print("G6 text is a line on x with y = z = 0; the echo, reading and reconstruction "
          "tests are run in step 8")


def test_g7_the_gates_differ_in_declared_widths_alone():
    before = _before()["declared"]
    for name in ("MM_xor.xml",):
        now, then = _declared(os.path.join(_DATA, name)), before[name]
        changed = {(s, k) for s, v in then.items() for k in v if now[s][k] != v[k]}
        assert changed and all(k in _WIDTHS for _s, k in changed)
    with open(os.path.join(_PROJECT, "test", "test_orders_learned.py"), encoding="utf-8") as fh:
        assert "_cs(nS=128, order=3, dim=24)" in fh.read()
    print("G7 MM_xor.xml differs in declared widths alone; the item 2.5 gate's fixture "
          "(equality_fixture) is 24 wide for 16: its band grew by 8")


def test_g8_the_great_registry_is_made_without_its_grid():
    from WhereRegistry import WhereRegistry
    great = WhereRegistry((("text", GREATEST),))
    great.place(torch.tensor([0, GREATEST - 1]))
    assert great.capacity == GREATEST and "grid" not in great.__dict__
    print(f"G8 a registry of {GREATEST} addresses is made, and places read, with no grid built")


def _least(wires, pairs=True):
    neighbour = float((wires[1:] - wires[:-1]).norm(dim=-1).min())
    if not pairs:
        return neighbour, None
    least = math.inf
    for lo in range(0, len(wires), 1024):
        d = torch.cdist(wires[lo:lo + 1024], wires)
        d[torch.arange(len(d)), torch.arange(lo, lo + len(d))] = math.inf
        least = min(least, float(d.min()))
    return neighbour, least


def test_g9_four_wires_or_three(xor_model):
    """A measure: the least distance between the wires of two neighbouring
    places, and between any two places, on one axis. Four wires: the
    two-rung ladder. Three, in the forms that fit: three phases of the long
    rung alone, and three phases of the short rung alone."""
    from Spaces import _ladder_periods, unsigned_phase
    lines = []
    for capacity in (xor_model.where_registry.capacity, 1 << 16):
        enc = _one_axis(capacity)
        x = torch.arange(capacity)
        four = enc.encode(x).double()
        long, short = _ladder_periods(capacity)
        shift = x + enc.where_origin
        def phases(period):
            angle = (shift % period).double() * 2 * math.pi / period
            return unsigned_phase(torch.stack([torch.cos(angle + k * 2 * math.pi / 3) for k in range(3)], -1))
        for form, wires in (("four, two rungs", four), ("three, long rung", phases(long)),
                            ("three, short rung", phases(short))):
            n, p = _least(wires, pairs=capacity < 1 << 14)
            lines.append(f"capacity {capacity} {form}: neighbours {n:.3e}"
                         + ("" if p is None else f", any two {p:.3e}"))
    for line in lines:
        print("G9", line)
