"""Chạy trọn một thí nghiệm: (huấn luyện) -> suy luận -> chấm bằng scripts/score.py.

Mỗi thí nghiệm chạy trong một tiến trình riêng để giải phóng VRAM khi xong. Có thể chạy lại
cùng lệnh sau khi Colab ngắt: huấn luyện tiếp từ checkpoint, suy luận tiếp phần còn thiếu.

Ví dụ:
  python -m lab.run --config configs/e0.yaml --name e0
  python -m lab.run --config configs/e1.yaml --name e1
  python -m lab.run --config configs/e1.yaml --name debug --max-steps 5 --split val --limit 40
  python -m lab.run --config configs/e2.yaml --name e4_data_size_0.1 --train-fraction 0.1 --label "E4 data_size=0.1"
"""
import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml

from lab.common import (ROOT, apply_setup, collate, encode_example, load_model, load_split,
                        load_trainable, make_prompt, objects_to_answer, open_image, predict,
                        trainable_state)
from scripts.labcommon import write_jsonl


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True, help="tên thư mục work/<name> và report/evidence/<name>.json")
    ap.add_argument("--label", help="tên hiển thị trong file bằng chứng (mặc định: name viết hoa)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, help="chỉ suy luận N ảnh đầu (gỡ lỗi; chấm trên đúng N ảnh đó)")
    # các biến cho E4 (ghi đè config)
    ap.add_argument("--r", type=int, help="LoRA rank (alpha = 2*r)")
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--coord", choices=["norm1000", "pixel"], default="norm1000")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--infer-batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--skip-infer", action="store_true")
    return ap.parse_args(argv)


def resolve_cfg(a):
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if "lora" in cfg:
        lora = dict(cfg["lora"])
        if a.r:
            lora.update(r=a.r, alpha=2 * a.r)
        cfg["lora"] = lora
        for k in ("max_steps", "batch_size", "grad_accum"):
            if getattr(a, k) is not None:
                cfg[k] = getattr(a, k)
    cfg.update(train_fraction=a.train_fraction, coord=a.coord)
    return cfg


def micro_batches(n_items, bs, seed):
    """Dòng chỉ số vô hạn, xáo lại mỗi epoch, xác định theo seed (để resume đúng chỗ)."""
    epoch = 0
    while True:
        idx = list(range(n_items))
        random.Random(seed * 1000 + epoch).shuffle(idx)
        for i in range(0, n_items - bs + 1, bs):
            yield idx[i:i + bs]
        epoch += 1


def train(model, processor, cfg, run_dir, data_dir, prompt):
    import torch
    from transformers import get_cosine_schedule_with_warmup

    entries = load_split("train", data_dir)
    if cfg["train_fraction"] < 1.0:
        k = max(1, round(len(entries) * cfg["train_fraction"]))
        entries = random.Random(cfg["seed"]).sample(entries, k)
    print(f"train: {len(entries)} ảnh")

    steps, bs, accum = cfg["max_steps"], cfg["batch_size"], cfg["grad_accum"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    scaler = torch.amp.GradScaler("cuda")
    pad_id = processor.tokenizer.pad_token_id

    ckpt_path = run_dir / "ckpt.pt"
    start, prev_minutes, prev_peak = 0, 0.0, 0.0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["trainable"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, prev_minutes, prev_peak = ck["step"], ck["minutes"], ck["peak_vram_gb"]
        print(f"resume từ bước {start}")

    def save_ckpt(step, minutes, peak):
        tmp = run_dir / "ckpt.tmp"
        torch.save({"step": step, "trainable": trainable_state(model), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "minutes": minutes, "peak_vram_gb": peak}, tmp)
        tmp.replace(ckpt_path)
        model.save_pretrained(run_dir / "adapter")  # adapter LoRA (projector KHÔNG nằm trong này)

    stream = micro_batches(len(entries), bs, cfg["seed"])
    for _ in range(start * accum):  # bỏ qua các batch đã học trước khi bị ngắt
        next(stream)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    t0 = time.time()
    log_f = open(run_dir / "train_log.jsonl", "a", encoding="utf-8")
    running = []
    for step in range(start, steps):
        step_loss = 0.0
        for _ in range(accum):
            feats = []
            for i in next(stream):
                e = entries[i]
                ans = objects_to_answer(e["objects"], cfg["coord"], e["width"], e["height"])
                feats.append(encode_example(processor, open_image(e, data_dir), prompt, ans))
            batch = {k: v.to(model.device) for k, v in collate(feats, pad_id).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step} (tràn fp16?) -> giảm lr / batch_size=1")
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()

        running.append(step_loss)
        log_f.write(json.dumps({"step": step + 1, "loss": step_loss, "lr": sched.get_last_lr()[0]}) + "\n")
        log_f.flush()
        minutes = prev_minutes + (time.time() - t0) / 60
        peak = max(prev_peak, torch.cuda.max_memory_allocated() / 1024 ** 3)
        if (step + 1) % 10 == 0 or step == start:
            print(f"bước {step + 1}/{steps}  loss={sum(running) / len(running):.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  {minutes:.1f} phút  VRAM đỉnh {peak:.2f} GB", flush=True)
            running = []
        if (step + 1) % cfg["save_every"] == 0 and step + 1 < steps:
            save_ckpt(step + 1, minutes, peak)
            print(f"  đã lưu checkpoint bước {step + 1}")
    log_f.close()

    minutes = prev_minutes + (time.time() - t0) / 60
    peak = max(prev_peak, torch.cuda.max_memory_allocated() / 1024 ** 3)
    torch.save(trainable_state(model), run_dir / "trainable.pt")  # LoRA + projector (nếu có)
    model.save_pretrained(run_dir / "adapter")
    trainable = sum(p.numel() for p in params)
    run = {"trainable_params": trainable, "peak_vram_gb": round(peak, 2),
           "train_minutes": round(minutes, 1), "train_images": len(entries), "cfg": cfg}
    (run_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    if ckpt_path.exists():
        ckpt_path.unlink()
    del opt, scaler
    torch.cuda.empty_cache()
    print(f"xong huấn luyện: {run}")
    return run


def subset_splits(data_dir, split, ids, dst):
    """Tạo thư mục splits chỉ gồm các ảnh đã suy luận (để chấm gỡ lỗi với --limit)."""
    rows = [r for r in load_split(split, data_dir) if r["image_id"] in ids]
    write_jsonl(dst / f"{split}.jsonl", rows)
    (dst / "meta.json").write_text((Path(data_dir) / "splits" / "meta.json").read_text(encoding="utf-8"),
                                   encoding="utf-8")
    return dst


def score(pred, label, split, coord, out, splits_dir, meta):
    cmd = [sys.executable, str(ROOT / "scripts" / "score.py"), "--pred", str(pred), "--name", label,
           "--split", split, "--coord", coord, "--out", str(out), "--splits-dir", str(splits_dir)]
    for k, v in meta.items():
        cmd += ["--meta", f"{k}={v}"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv=None):
    a = parse_args(argv)
    cfg = resolve_cfg(a)
    cfg["save_every"] = a.save_every
    run_dir = Path(a.work_dir) / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = make_prompt(a.coord)
    trained = "lora" in cfg
    pred_path = run_dir / f"pred_{a.split}{'_' + str(a.limit) if a.limit else ''}.jsonl"
    entries = load_split(a.split, a.data_dir, a.limit)
    need_infer = not a.skip_infer and not (
        pred_path.exists() and len(pred_path.read_text(encoding="utf-8").splitlines()) >= len(entries))
    need_train = trained and not (run_dir / "trainable.pt").exists()

    if need_train or need_infer:
        import torch
        torch.manual_seed(cfg.get("seed", 42))
        model, processor = load_model(cfg.get("model_id"))
        if trained:
            lora = cfg["lora"]
            model, info = apply_setup(model, lora["targets"], cfg["train_projector"], lora["r"],
                                      lora["alpha"], lora["dropout"], gradient_checkpointing=need_train)
            (run_dir / "setup.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            if need_train:
                train(model, processor, cfg, run_dir, a.data_dir, prompt)
            else:
                load_trainable(model, run_dir / "trainable.pt")
            model.config.use_cache = True
        if need_infer:
            predict(model, processor, entries, prompt, pred_path, a.data_dir,
                    a.infer_batch, a.max_new_tokens)
    if a.skip_infer:
        return

    meta = {}
    if trained:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        meta = {k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")}
    label = a.label or a.name.upper()
    splits_dir = Path(a.data_dir) / "splits"
    if a.split == "test" and not a.limit:
        out = ROOT / "report" / "evidence" / f"{a.name}.json"
    else:
        out = run_dir / f"score_{pred_path.stem}.json"
        if a.limit:
            splits_dir = subset_splits(a.data_dir, a.split, {e["image_id"] for e in entries},
                                       run_dir / f"splits_{a.split}_{a.limit}")
    score(pred_path, label, a.split, a.coord, out, splits_dir, meta)


if __name__ == "__main__":
    main()
