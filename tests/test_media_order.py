from __future__ import annotations

# Ordering checks for the media references on StrataRun. Nothing here starts or stops a
# Strata instance: only the socket-ordering helpers and the bypass path are exercised, so
# these tests run with no GPU and no server resident.

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _fake_comfy_modules() -> dict[str, types.ModuleType]:
    """A stub comfy_api.latest so strata.py imports without ComfyUI present."""
    io_mod = types.ModuleType("comfy_api.io")

    class NodeOutput:
        def __init__(self, *values):
            self.values = values

    io_mod.NodeOutput = NodeOutput
    for name in ("ComfyNode", "Schema", "String", "Int", "Float",
                 "Boolean", "Combo", "Image", "Audio", "Autogrow"):
        setattr(io_mod, name, object)

    latest = types.ModuleType("comfy_api.latest")
    latest.ComfyExtension = object
    latest.io = io_mod
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest
    return {"comfy_api": comfy_api, "comfy_api.latest": latest, "comfy_api.io": io_mod}


def import_strata():
    package = types.ModuleType("strata_testpkg")
    package.__path__ = [str(PROJECT_ROOT)]

    patched = _fake_comfy_modules()
    patched["strata_testpkg"] = package
    with mock.patch.dict(sys.modules, patched):
        return importlib.import_module("strata_testpkg.strata")


class OrderingTests(unittest.TestCase):
    def test_socket_indexes_are_ordered(self) -> None:
        module = import_strata()
        pictures = module.StrataRun._ordered_pictures(
            {"ref_image_2": "c", "ref_image_0": "a", "ref_image_1": "b"}, None, False
        )
        self.assertEqual(pictures, ["a", "b", "c"])

    def test_empty_slots_are_skipped(self) -> None:
        module = import_strata()
        pictures = module.StrataRun._ordered_pictures(
            {"ref_image_0": "a", "ref_image_1": None, "ref_image_3": "d"}, None, False
        )
        self.assertEqual(pictures, ["a", "d"])

    def test_first_frame_only_joins_when_asked(self) -> None:
        module = import_strata()
        without = module.StrataRun._ordered_pictures({"ref_image_0": "a"}, "first", False)
        with_frame = module.StrataRun._ordered_pictures({"ref_image_0": "a"}, "first", True)
        self.assertEqual(without, ["a"])
        # appended after the user refs so it takes the next Picture ordinal, like the VL node
        self.assertEqual(with_frame, ["a", "first"])

    def test_last_frame_is_never_a_picture(self) -> None:
        module = import_strata()
        pictures = module.StrataRun._ordered_pictures({"ref_image_0": "a"}, None, False)
        self.assertEqual(pictures, ["a"])

    def test_batches_expand_in_order(self) -> None:
        module = import_strata()
        seen: list[object] = []
        module._image_part = lambda image: seen.append(image) or {"type": "image_url"}

        class Batch:  # a (B,H,W,C) IMAGE: ndim 4, iterating yields one frame per picture
            ndim = 4
            frames = ["a", "b"]

            def __iter__(self):
                return iter(self.frames)

        parts = module._picture_parts([Batch(), None, "single"])
        self.assertEqual(len(parts), 3)
        self.assertEqual(seen, ["a", "b", "single"])


class BypassTests(unittest.TestCase):
    def test_bypass_never_touches_strata(self) -> None:
        module = import_strata()
        out = module.StrataRun.execute(
            folder="", port=8080, prompt="hello", bypass=True,
            ref_images={"ref_image_0": object()},
        )
        self.assertEqual(out.values[0], "hello")


if __name__ == "__main__":
    unittest.main()
