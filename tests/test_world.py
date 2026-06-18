import json

import pytest

from gestalt.world import World, WorldError


def test_add_and_get():
    w = World()
    snap = w.add("box", kind="cube", x=1, y=2, z=0, color="red")
    assert snap["name"] == "box" and snap["kind"] == "cube" and snap["color"] == "red"
    assert w.get("box")["x"] == 1
    assert w.names() == ["box"]


def test_duplicate_name_rejected():
    w = World()
    w.add("box")
    with pytest.raises(WorldError):
        w.add("box")


def test_missing_object_errors():
    w = World()
    with pytest.raises(WorldError):
        w.get("nope")
    with pytest.raises(WorldError):
        w.move("nope", 1, 0, 0)


def test_move_and_place():
    w = World()
    w.add("box", x=0, y=0, z=0)
    w.move("box", dx=1, dy=2, dz=3)
    o = w.get("box")
    assert (o["x"], o["y"], o["z"]) == (1, 2, 3)
    w.place("box", 5, 5, 5)
    o = w.get("box")
    assert (o["x"], o["y"], o["z"]) == (5, 5, 5)


def test_distance_and_find():
    w = World()
    w.add("a", kind="cube", color="red", x=0, y=0, z=0)
    w.add("b", kind="cube", color="blue", x=3, y=4, z=0)
    assert w.distance("a", "b") == pytest.approx(5.0)
    assert {o["name"] for o in w.find(kind="cube")} == {"a", "b"}
    assert [o["name"] for o in w.find(color="blue")] == ["b"]
    assert [o["name"] for o in w.find(near="a", radius=10)] == ["b"]
    assert w.find(near="a", radius=1) == []


def test_remove_and_clear():
    w = World()
    w.add("a")
    w.add("b")
    w.remove("a")
    assert w.names() == ["b"]
    assert w.clear() == 1
    assert w.names() == []


def test_file_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "world.json")
    w = World(path=path)
    w.add("box", kind="cube", x=1, y=2, z=3, color="green")
    # A second World pointed at the same file sees the saved state.
    w2 = World(path=path)
    assert w2.get("box")["color"] == "green"
    on_disk = json.loads((tmp_path / "world.json").read_text())
    assert on_disk["objects"][0]["name"] == "box"


def test_scene_text():
    w = World()
    assert "empty" in w.scene_text().lower()
    w.add("box", kind="cube", color="red", x=1, y=0, z=0)
    assert "box" in w.scene_text() and "cube" in w.scene_text()
