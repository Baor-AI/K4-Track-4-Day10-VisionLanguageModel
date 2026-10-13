"""Reward IoU cho GRPO (E5). Python thuần, test được không cần GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score import IOU_THR, iou_xyxy, parse_output  # noqa: E402


def gt_boxes(objects):
    """objects dạng split jsonl (yolo) -> [{"label", "bbox"}] thang 0–1000."""
    out = []
    for o in objects:
        xc, yc, w, h = o["yolo"]
        out.append({"label": o["label"], "bbox": ((xc - w / 2) * 1000, (yc - h / 2) * 1000,
                                                  (xc + w / 2) * 1000, (yc + h / 2) * 1000)})
    return out


def reward(completion, gt):
    """0 nếu JSON hỏng; 1 nếu cả gt và pred rỗng; còn lại 0.2 + 0.8 * tổng IoU khớp / max(n_gt, n_pred)."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt and not pred:
        return 1.0
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for v, i, j in pairs:
        if v < IOU_THR:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += v
    return 0.2 + 0.8 * total / max(len(gt), len(pred), 1)
