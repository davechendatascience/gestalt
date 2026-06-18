"""The sandbox world: a small in-memory scene of named 3D objects.

This module is pure logic with no MCP / network dependencies, so it is trivially
testable and reusable. The MCP server in ``gestalt.mcp_server`` wraps a single
``World`` instance and exposes its methods as tools.

Optionally the world is mirrored to a JSON file (set ``path`` or the
``GESTALT_WORLD_FILE`` env var) so an external process — e.g. a notebook — can
read and visualize the scene after the agent has acted on it.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict, dataclass


class WorldError(Exception):
    """Raised for invalid operations (duplicate names, missing objects, ...)."""


@dataclass
class WorldObject:
    name: str
    kind: str = "thing"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    color: str | None = None
    notes: str | None = None

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class World:
    """An in-memory scene of uniquely-named objects, optionally file-backed."""

    def __init__(self, path: str | None = None):
        self._objs: dict[str, WorldObject] = {}
        self._lock = threading.RLock()
        self._path = path
        if path and os.path.exists(path):
            self.load(path)

    # ----- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"objects": [asdict(o) for o in self._objs.values()]}

    def load(self, path: str) -> None:
        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._objs = {
                o["name"]: WorldObject(**o) for o in data.get("objects", [])
            }

    def _save(self) -> None:
        if not self._path:
            return
        tmp = f"{self._path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, self._path)

    # ----- helpers -----------------------------------------------------------
    def _require(self, name: str) -> WorldObject:
        obj = self._objs.get(name)
        if obj is None:
            known = ", ".join(sorted(self._objs)) or "(none)"
            raise WorldError(f"No object named {name!r}. Known objects: {known}.")
        return obj

    # ----- mutations ---------------------------------------------------------
    def add(
        self,
        name: str,
        kind: str = "thing",
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        color: str | None = None,
        notes: str | None = None,
    ) -> dict:
        with self._lock:
            if name in self._objs:
                raise WorldError(f"An object named {name!r} already exists.")
            obj = WorldObject(name, kind, float(x), float(y), float(z), color, notes)
            self._objs[name] = obj
            self._save()
            return asdict(obj)

    def place(self, name: str, x: float, y: float, z: float) -> dict:
        """Move an object to an absolute position."""
        with self._lock:
            obj = self._require(name)
            obj.x, obj.y, obj.z = float(x), float(y), float(z)
            self._save()
            return asdict(obj)

    def move(self, name: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict:
        """Translate an object by a relative offset."""
        with self._lock:
            obj = self._require(name)
            obj.x += float(dx)
            obj.y += float(dy)
            obj.z += float(dz)
            self._save()
            return asdict(obj)

    def set_color(self, name: str, color: str) -> dict:
        with self._lock:
            obj = self._require(name)
            obj.color = color
            self._save()
            return asdict(obj)

    def remove(self, name: str) -> dict:
        with self._lock:
            obj = self._require(name)
            del self._objs[name]
            self._save()
            return asdict(obj)

    def clear(self) -> int:
        with self._lock:
            n = len(self._objs)
            self._objs.clear()
            self._save()
            return n

    # ----- queries -----------------------------------------------------------
    def get(self, name: str) -> dict:
        with self._lock:
            return asdict(self._require(name))

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._objs)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [asdict(o) for o in self._objs.values()]

    def distance(self, a: str, b: str) -> float:
        with self._lock:
            oa, ob = self._require(a), self._require(b)
            return math.dist(oa.position, ob.position)

    def find(
        self,
        kind: str | None = None,
        color: str | None = None,
        near: str | None = None,
        radius: float | None = None,
    ) -> list[dict]:
        """Filter objects by kind/color and/or proximity to another object."""
        with self._lock:
            anchor = self._require(near) if near else None
            out = []
            for obj in self._objs.values():
                if kind is not None and obj.kind != kind:
                    continue
                if color is not None and obj.color != color:
                    continue
                if anchor is not None:
                    if obj.name == anchor.name:
                        continue
                    if radius is not None and math.dist(obj.position, anchor.position) > radius:
                        continue
                out.append(asdict(obj))
            return out

    def scene_text(self) -> str:
        with self._lock:
            if not self._objs:
                return "The world is empty."
            lines = [f"{len(self._objs)} object(s) in the world:"]
            for o in self._objs.values():
                color = f"{o.color} " if o.color else ""
                note = f" — {o.notes}" if o.notes else ""
                lines.append(
                    f"  - {o.name}: {color}{o.kind} at "
                    f"({o.x:g}, {o.y:g}, {o.z:g}){note}"
                )
            return "\n".join(lines)
