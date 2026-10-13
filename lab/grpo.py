"""E5 (tuỳ chọn): GRPO tối giản với reward IoU, khởi tạo từ setup tốt nhất của E1–E3.

Mỗi bước: lấy 1 ảnh train, sinh G completion bằng sampling, chấm reward, chuẩn hoá thành advantage
trong nhóm, rồi cập nhật policy gradient trên log-prob của completion. Một cập nhật cho mỗi nhóm nên
tỉ số importance = 1 (không cần clip), beta = 0 nên không cần model tham chiếu (theo configs/e5.yaml).
Viết tay thay vì trl.GRPOTrainer để chạy được trên T4 với model 4-bit đã có projector fp32.

  python -m lab.grpo --base e2 --name e5 --data-dir /content/data
"""
import argparse
import json
import random
import time
from pathlib import Path

import yaml

from lab.common import (apply_setup, chat_prompt, load_model, load_split, load_trainable,
                        make_prompt, open_image, predict, trainable_state)
from lab.reward import gt_boxes, reward
from lab.run import score
from scripts.score import parse_output


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="tên run tốt nhất trong work/ (vd e2)")
    ap.add_argument("--name", default="e5")
    ap.add_argument("--config", default="configs/e5.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=384, help="cho suy luận test sau GRPO")
    a = ap.parse_args(argv)

    import torch

    g = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))["grpo"]
    G, C = g["num_generations"], g["max_completion_length"]
    steps = a.max_steps or g["max_steps"]
    base_dir = Path(a.work_dir) / a.base
    base = json.loads((base_dir / "run.json").read_text(encoding="utf-8"))["cfg"]
    run_dir = Path(a.work_dir) / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = make_prompt(base.get("coord", "norm1000"))

    torch.manual_seed(0)
    model, processor = load_model(base.get("model_id"))
    lora = base["lora"]
    model, info = apply_setup(model, lora["targets"], base["train_projector"], lora["r"], lora["alpha"],
                              lora["dropout"])
    load_trainable(model, base_dir / "trainable.pt")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=g["learning_rate"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    tok = processor.tokenizer
    stop_ids = torch.tensor(sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id,
                                    *([tok.eos_token_id] if tok.eos_token_id is not None else [])}))

    entries = [e for e in load_split("train", a.data_dir)]
    rng = random.Random(0)
    ptext = chat_prompt(processor, prompt)
    log_f = open(run_dir / "grpo_log.jsonl", "w", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(steps):
        e = rng.choice(entries)
        gt = gt_boxes(e["objects"])
        img = open_image(e, a.data_dir)
        inputs = processor(text=[ptext] * G, images=[img] * G, return_tensors="pt").to(model.device)
        P = inputs["input_ids"].shape[1]

        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(**inputs, do_sample=True, temperature=a.temperature,
                                 max_new_tokens=C, use_cache=True)
        comp = out[:, P:]
        texts = processor.batch_decode(comp, skip_special_tokens=True)
        rewards = torch.tensor([reward(t, gt) for t in texts], dtype=torch.float32)
        parsed = [parse_output(t) for t in texts]
        valid = [p for p in parsed if p is not None]
        areas = [(b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) / 1e6
                 for p in valid for b in p]
        rec = {"step": step + 1, "reward_mean": rewards.mean().item(), "reward_std": rewards.std().item(),
               "valid_frac": len(valid) / G, "empty_frac": sum(1 for p in valid if not p) / G,
               "mean_pred_boxes": sum(len(p) for p in valid) / max(len(valid), 1),
               "mean_box_area": sum(areas) / max(len(areas), 1), "n_gt": len(gt),
               "comp_len": comp.shape[1], "updated": False}

        if rewards.std() > 1e-6:
            adv = ((rewards - rewards.mean()) / (rewards.std() + 1e-4)).to(model.device)
            # mask: token completion tới và gồm token dừng đầu tiên
            is_stop = torch.isin(comp, stop_ids.to(comp.device))
            after = (is_stop.cumsum(1) - is_stop.long()) > 0
            mask = (~after).float()
            attn = torch.cat([inputs["attention_mask"], mask.long()], dim=1)
            fwd = {"input_ids": out, "attention_mask": attn, "pixel_values": inputs["pixel_values"],
                   "image_grid_thw": inputs["image_grid_thw"]}
            if "mm_token_type_ids" in inputs:
                fwd["mm_token_type_ids"] = torch.cat(
                    [inputs["mm_token_type_ids"], torch.zeros_like(comp, dtype=inputs["mm_token_type_ids"].dtype)], 1)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**fwd, logits_to_keep=comp.shape[1] + 1).logits[:, :-1]
            logp = torch.log_softmax(logits.float(), -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
            seq_logp = (logp * mask).sum(1) / mask.sum(1).clamp(min=1)
            loss = -(adv * seq_logp).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            rec.update(loss=loss.item(), updated=True)
            del logits, logp
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if (step + 1) % 5 == 0 or step == 0:
            print(f"bước {step + 1}/{steps} reward={rec['reward_mean']:.3f}±{rec['reward_std']:.3f} "
                  f"valid={rec['valid_frac']:.2f} rỗng={rec['empty_frac']:.2f} "
                  f"box/ảnh={rec['mean_pred_boxes']:.1f} diện tích={rec['mean_box_area']:.3f}", flush=True)
    log_f.close()

    minutes = (time.time() - t0) / 60
    torch.save(trainable_state(model), run_dir / "trainable.pt")
    run = {"trainable_params": sum(p.numel() for p in params),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
           "train_minutes": round(minutes, 1), "base": a.base, "grpo": g, "steps": steps,
           "cfg": base}
    (run_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    opt.zero_grad(set_to_none=True)
    del opt
    torch.cuda.empty_cache()

    model.config.use_cache = True
    entries = load_split("test", a.data_dir)
    pred = predict(model, processor, entries, prompt, run_dir / "pred_test.jsonl", a.data_dir,
                   4, a.max_new_tokens)
    from lab.common import ROOT
    score(pred, a.name.upper(), "test", base.get("coord", "norm1000"),
          ROOT / "report" / "evidence" / f"{a.name}.json", Path(a.data_dir) / "splits",
          {k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")})


if __name__ == "__main__":
    main()
