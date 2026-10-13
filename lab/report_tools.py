"""Công cụ báo cáo: bảng tổng hợp từ report/evidence, ảnh minh hoạ đúng/sai, đường loss."""
import json
from pathlib import Path

from lab.common import ROOT, load_split, open_image
from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output

SETUPS = {"e0": "zero-shot", "e1": "LoRA LLM", "e2": "+ projector", "e3": "+ LoRA ViT", "e5": "GRPO"}
COLORS = {"pothole": (255, 64, 64), "crack": (255, 200, 0), "manhole": (0, 160, 255)}


def summary_table(evidence_dir=ROOT / "report" / "evidence"):
    """Các dòng markdown đúng cột của bảng trong REPORT.md."""
    lines = ["| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh "
             "| F1 pothole | F1 crack | F1 manhole |", "|" + "---|" * 11]
    for p in sorted(Path(evidence_dir).glob("e*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        m, pc, meta = d["micro"], d["per_class"], d.get("meta", {})
        lines.append("| {} | {} | {} | {} | {} | {:.3f} | {:.3f}–{:.3f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} |".format(
            d["name"], SETUPS.get(p.stem, p.stem), f"{meta['trainable_params']:,}" if "trainable_params" in meta else 0,
            meta.get("peak_vram_gb", "–"), meta.get("train_minutes", "–"), m["f1"], *d["f1_ci95"],
            d["fp_per_image"], *(pc[c]["f1"] for c in CLASSES)))
    return "\n".join(lines)


def _per_image(pred_path, split, data_dir, coord):
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(pred_path)}
    rows = []
    for e in load_split(split, data_dir):
        if e["image_id"] not in preds:
            continue
        parsed = parse_output(preds[e["image_id"]])
        pred = _to_norm1000(parsed or [], coord, e["width"], e["height"])
        per, _ = _match(_gt_boxes(e), pred)
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        rows.append({"entry": e, "pred": pred, "valid": parsed is not None, "tp": tp, "fp": fp, "fn": fn,
                     "raw": preds[e["image_id"]]})
    return rows


def draw(entry, pred, data_dir, title=""):
    from PIL import ImageDraw
    img = open_image(entry, data_dir)
    W, H = img.size
    d = ImageDraw.Draw(img)
    for g in _gt_boxes(entry):  # box đúng: xanh lá, nét dày
        x1, y1, x2, y2 = g["bbox"]
        d.rectangle([x1 * W / 1000, y1 * H / 1000, x2 * W / 1000, y2 * H / 1000], outline=(0, 255, 0), width=3)
    for p in pred:  # box dự đoán: màu theo lớp, nét mảnh + nhãn
        x1, y1, x2, y2 = p["bbox"]
        c = COLORS.get(p["label"], (255, 0, 255))
        d.rectangle([x1 * W / 1000, y1 * H / 1000, x2 * W / 1000, y2 * H / 1000], outline=c, width=2)
        d.text((x1 * W / 1000 + 2, y1 * H / 1000 + 2), p["label"], fill=c)
    if title:
        d.rectangle([0, 0, W, 16], fill=(0, 0, 0))
        d.text((4, 2), title, fill=(255, 255, 255))
    return img


def save_examples(name, data_dir="data", split="test", coord="norm1000", n_good=3, n_bad=3,
                  work_dir=ROOT / "work", out_dir=ROOT / "report" / "figures"):
    """Lưu n_good ảnh đúng nhiều nhất + n_bad ảnh sai nhiều nhất của một run. Trả list đường dẫn."""
    rows = _per_image(Path(work_dir) / name / f"pred_{split}.jsonl", split, data_dir, coord)
    good = sorted([r for r in rows if r["tp"] and not r["fp"] and not r["fn"]], key=lambda r: -r["tp"])[:n_good]
    bad = sorted([r for r in rows if r["fp"] or r["fn"] or not r["valid"]], key=lambda r: -(r["fp"] + r["fn"]))[:n_bad]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tag, group in (("dung", good), ("sai", bad)):
        for r in group:
            t = f"{name} {r['entry']['image_id']} TP={r['tp']} FP={r['fp']} FN={r['fn']}" + ("" if r["valid"] else " INVALID JSON")  # font mặc định của PIL chỉ có ASCII
            p = out_dir / f"{name}_{tag}_{r['entry']['image_id']}.jpg"
            draw(r["entry"], r["pred"], data_dir, t).save(p, quality=85)
            paths.append(p)
    return paths


def plot_losses(names, work_dir=ROOT / "work", out=ROOT / "report" / "figures" / "loss.png"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        p = Path(work_dir) / n / "train_log.jsonl"
        if not p.exists():
            continue
        rows = read_jsonl(p)
        last = {r["step"]: r["loss"] for r in rows}  # nếu resume, giữ giá trị cuối mỗi bước
        xs = sorted(last)
        ys = [last[x] for x in xs]
        k = 10
        sm = [sum(ys[max(0, i - k + 1):i + 1]) / len(ys[max(0, i - k + 1):i + 1]) for i in range(len(ys))]
        ax.plot(xs, sm, label=n)
    ax.set_xlabel("bước tối ưu")
    ax.set_ylabel("loss (trung bình trượt 10 bước)")
    ax.legend()
    ax.grid(alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    return fig
