import json

import pytest

from lab.common import objects_to_answer, yolo_to_box
from lab.reward import reward

GT = [{"label": "pothole", "bbox": (100.0, 100.0, 300.0, 300.0)},
      {"label": "crack", "bbox": (500.0, 500.0, 700.0, 600.0)}]


def _ans(*boxes):
    return json.dumps([{"label": lab, "bbox_2d": list(bb)} for lab, bb in boxes])


def test_perfect():
    assert reward(_ans(("pothole", (100, 100, 300, 300)), ("crack", (500, 500, 700, 600))), GT) == pytest.approx(1.0)


def test_broken_json():
    assert reward('[{"label": "pothole", "bbox_2d": [1, 2', GT) == 0.0
    assert reward("không có gì", GT) == 0.0


def test_wrong_class_gets_only_format_reward():
    assert reward(_ans(("manhole", (100, 100, 300, 300)), ("manhole", (500, 500, 700, 600))), GT) == pytest.approx(0.2)


def test_extra_box_is_penalised():
    good = _ans(("pothole", (100, 100, 300, 300)), ("crack", (500, 500, 700, 600)))
    extra = _ans(("pothole", (100, 100, 300, 300)), ("crack", (500, 500, 700, 600)), ("crack", (0, 0, 50, 50)))
    assert reward(extra, GT) < reward(good, GT)
    assert reward(extra, GT) == pytest.approx(0.2 + 0.8 * 2 / 3)


def test_empty():
    assert reward("[]", []) == 1.0
    assert reward("[]", GT) == pytest.approx(0.2)
    assert reward(_ans(("crack", (0, 0, 10, 10))), []) == pytest.approx(0.2)


def test_giant_box_does_not_hack():
    assert reward(_ans(("pothole", (0, 0, 1000, 1000))), GT) == pytest.approx(0.2)


def test_low_iou_not_matched():
    assert reward(_ans(("pothole", (250, 250, 450, 450)), ("crack", (500, 500, 700, 600))), GT) == pytest.approx(0.2 + 0.8 * 1 / 2)


def test_fenced_json_is_parsed():
    raw = "```json\n" + _ans(("pothole", (100, 100, 300, 300)), ("crack", (500, 500, 700, 600))) + "\n```"
    assert reward(raw, GT) == pytest.approx(1.0)


def test_yolo_to_box_and_order():
    assert yolo_to_box("crack", [0.5, 0.5, 0.2, 0.4]) == {"label": "crack", "bbox_2d": [400, 300, 600, 700]}
    assert yolo_to_box("crack", [0.5, 0.5, 0.2, 0.4], "pixel", 640, 360)["bbox_2d"] == [256, 108, 384, 252]
    assert yolo_to_box("crack", [0.01, 0.99, 0.1, 0.1])["bbox_2d"] == [0, 940, 60, 1000]  # kẹp biên
    objs = [{"label": "crack", "yolo": [0.5, 0.8, 0.1, 0.1]}, {"label": "pothole", "yolo": [0.2, 0.2, 0.1, 0.1]}]
    assert [b["label"] for b in json.loads(objects_to_answer(objs))] == ["pothole", "crack"]
    assert objects_to_answer([]) == "[]"
