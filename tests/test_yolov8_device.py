"""1.3: YOLOv8 forwards the configured device to inference.

The registry resolves a device from ``device_from_config`` and passes it to the
detector constructor, but the inference call used to drop it. These pin that the
device now reaches ultralytics when set, and that ``None`` still means
auto-select (the prior, unchanged behaviour) rather than ``device=None`` noise.
"""

from __future__ import annotations

from cinemateca.models.objects.yolov8 import YOLOv8ObjectDetector


class _FakeResult:
    boxes: list = []


class _FakeModel:
    names: dict = {}

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return [_FakeResult()]


def _detector(device):
    det = YOLOv8ObjectDetector(cfg=None, device=device)
    det._model = _FakeModel()  # pre-seed so _load_model short-circuits (no ultralytics import)
    return det


def test_detect_forwards_device_when_set():
    det = _detector("cuda")
    det.detect("frame.jpg")
    _src, kwargs = det._model.calls[0]
    assert kwargs.get("device") == "cuda"
    assert kwargs["conf"] == 0.30
    assert kwargs["verbose"] is False


def test_detect_omits_device_when_none():
    det = _detector(None)
    det.detect("frame.jpg")
    _src, kwargs = det._model.calls[0]
    # None must not be forwarded — ultralytics auto-selects, matching the
    # behaviour before the fix (device kwarg simply absent).
    assert "device" not in kwargs
